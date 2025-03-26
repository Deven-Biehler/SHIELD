import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Lock, current_process
import numpy as np
import pandas as pd
import pickle
import time
from collections import defaultdict
from memory_profiler import profile, LogFile
import gc 
import psutil
from tqdm import tqdm  # Import tqdm for progress bar
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from utils import reset_directory, analyze_results, safe_write_to_csv

# Set log file for memory profiling output
log_file_path = "memory_profile.log"
log_file = open(log_file_path, "w+")
# sys.stdout = LogFile(log_file_path, reportIncrementFlag=False)

# read config file
with open('config.json', 'r') as file:
    config = json.load(file)

max_memory = config['max_memory'] * (1024 ** 3)  # Convert GB to bytes
max_memory = max_memory / 40
config['max_job_size'] = (max_memory) / config["max_workers"]  # Convert GB to bytes


max_workers = config['max_workers']

def memory_usage_within_limit():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss <= max_memory

def task_wrapper(task, *args, **kwargs):
    if not memory_usage_within_limit():
        raise MemoryError("Memory usage exceeded the limit set in config.json")
    return task(*args, **kwargs)

def chunk_file(num_chunks, file_size):
    chunk_size = file_size // num_chunks + 1
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(file_size, start + chunk_size)
        chunks.append((start, end - start))
    return chunks

def compute_partition_boundaries(file_name, child_boundaries, partition_columns):
    boundaries_dict = {}
    bin_sizes = []
    
    # Precompute boundaries & bin sizes
    for col in partition_columns:
        boundaries_dict[col] = [
            (child_boundaries[col][i], child_boundaries[col][i + 1])
            for i in range(len(child_boundaries[col]) - 1)
        ]
        bin_sizes.append(len(boundaries_dict[col]))

    # Calculate all possible combinations of indices for the partition columns
    indices = np.indices(bin_sizes).reshape(len(bin_sizes), -1).T

    # Compute and store the boundaries for each partition index
    partitions = {}
    for idx_array in indices:
        bin_indices = tuple(idx_array)
        boundaries = {
            col: boundaries_dict[col][idx]
            for col, idx in zip(partition_columns, bin_indices)
        }
        # Compute a unique partition index using np.ravel_multi_index if needed later
        partition_index = np.ravel_multi_index(bin_indices, dims=bin_sizes, mode='clip')
        original_index = os.path.basename(file_name).split(".")[0] + "-"
        partitions[original_index + f"{partition_index}"] = boundaries

    return partitions


# @profile(stream=open("memory_profile.log", "a+"))
def _partition_data(df, child_boundaries, partition_columns):
    # print(df.columns)
    boundaries_dict = {}
    bin_values = []

    # Step 1: Precompute boundaries & bin arrays
    for col in partition_columns:
        boundaries_dict[col] = [
            (child_boundaries[col][i], child_boundaries[col][i + 1])
            for i in range(len(child_boundaries[col]) - 1)
        ]
        # pd.cut -> integer bin labels
        bin_labels = pd.cut(
            df[col],
            bins=child_boundaries[col],
            labels=False,
            include_lowest=True
        ).to_numpy()

        # Convert NaN -> -1
        bin_values.append(np.nan_to_num(bin_labels, nan=-1).astype(int))

    # Step 2: Compute partition_index (optionally using manual method from #3)
    bin_sizes = [len(child_boundaries[col]) - 1 for col in partition_columns]
    partition_index = np.ravel_multi_index(
        np.column_stack(bin_values).T,
        dims=bin_sizes,
        mode='clip'
    )
    
    # Step 3: Group and construct partitions
    partitions = {}
    grouped = df.groupby(pd.Series(partition_index, index=df.index), sort=False)
    del df

    for grp_partition_index, subset in grouped:
        bin_indices = np.unravel_index(grp_partition_index, bin_sizes)
        boundaries = {
            col: boundaries_dict[col][idx]
            for col, idx in zip(partition_columns, bin_indices)
        }

        partitions[grp_partition_index] = {
            'data': subset,
            'boundaries': boundaries
        }

    for partition in partitions:
        if "data" not in partitions[partition]:
            raise ValueError("Data not in partition")
    return partitions

def _calculate_child_boundaries(parent_boundary, partition_columns):
    child_boundaries = {}
    for i, col in enumerate(partition_columns):
        diff = parent_boundary[col][1] - parent_boundary[col][0]
        child_boundaries[col] = ((parent_boundary[col][0],
                                parent_boundary[col][0] + (diff / 3),
                                parent_boundary[col][0] + (diff / 3) * 2,
                                parent_boundary[col][1] + (diff / 3) * 3))
    return child_boundaries



import pandas as pd

def find_min_max(file_path, partition_columns):
    # Initialize dictionaries to store the global min and max values
    global_min = {}
    global_max = {}

    # Define the chunk size (number of rows per chunk)
    chunk_size = 10000000  # Adjust based on your system's memory capacity

    # Use the iterator to read the CSV in chunks
    i = 0
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        i += chunk_size
        # Calculate min and max for each column in the current chunk
        chunk_min = chunk.min()
        chunk_max = chunk.max()

        # Update global min and max values
        for column in partition_columns:
            # Update global minimum
            if column in global_min:
                global_min[column] = min(global_min[column], chunk_min[column])
            else:
                global_min[column] = chunk_min[column]

            # Update global maximum
            if column in global_max:
                global_max[column] = max(global_max[column], chunk_max[column])
            else:
                global_max[column] = chunk_max[column]

    return global_min, global_max


# @profile(stream=open("memory_profile.log", "a+"))
def _calculate_boundaries(file_path, partition_columns, global_min=None, global_max=None):
    if global_min is None or global_max is None:
        global_min, global_max = find_min_max(file_path, partition_columns)
    boundaries = {}
    for column in partition_columns:
        boundaries[column] = (global_min[column], global_max[column])
    # Find which column has the maximum range
    max_range = 0
    for column in partition_columns:
        boundary_range = boundaries[column][1] - boundaries[column][0]
        if boundary_range > max_range:
            max_range = boundary_range

    # Normalize the boundaries
    for column in partition_columns:
        boundary_range = abs(boundaries[column][1] - boundaries[column][0])
        diff = max_range - boundary_range
        # add diff / 2 to the lower bound and subtract diff / 2 from the upper bound
        boundaries[column] = (boundaries[column][0] - (diff / 2), boundaries[column][1] + (diff / 2))

    return boundaries

def write_pickle_dict(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def write_partition_data_to_csv(base_file, partitions, locks):
    base_file = os.path.basename(base_file).split(".")[0] + "-"
    for index, partition in partitions.items():
        new_path = os.path.join(config['tree_file'], base_file + str(index)+".csv")
        file_name = os.path.basename(new_path)
        try:
            lock = locks[file_name]
        except Exception as e:
            print(f"failed getting lock for {file_name}: {e}")
            print(new_path)

        data = partition["data"]
        
        safe_write_to_csv(new_path, data, lock)

@profile(stream=open("memory_profile.log", "a+"))
def worker(job):
    file_path, chunk, partition_columns, locks, child_boundaries = job
    start, size = chunk
    try:
        if start == 0:
            data = pd.read_csv(file_path, nrows=size)
        else:
            data = pd.read_csv(file_path, skiprows=start, nrows=size)
            data.columns = pd.read_csv(file_path, nrows=1).columns
    except Exception as e:
        print(f"file_path: {file_path}")
        print(f"start: {start}")
        print(f"size: {size}")
        print(f"Error reading csv file: {e}")
    try:
        partitions = _partition_data(data, child_boundaries, partition_columns)
    except Exception as e:
        print(f"Error partitioning data: {e}")
    del data
    write_partition_data_to_csv(file_path, partitions, locks)
    gc.collect()
            
def bytes_to_Gb(bytes):
    return bytes / (1024 ** 3)

def get_file_length(file_path):
    return sum(1 for line in open(file_path))

from concurrent.futures import ProcessPoolExecutor
import os

def exact_line_count(file_path):
    count = 0
    buffer_size = 1024 * 1024  # Read in chunks of 1 MB
    with open(file_path, 'rb') as file:
        while chunk := file.read(buffer_size):
            count += chunk.count(b'\n')
    return count



def generate_locks_from_partitions(manager, partitions, num_paritions):
    locks = {}
    for partition in partitions:
        index = os.path.basename(partition).split(".")[0]
        for i in range(num_paritions):
            locks[index + f"-{i}.csv"] = manager.Lock()
    return locks



def manage_jobs(jobs, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Initialize a list to hold references to job futures
        futures = []
        results = []

        # Submit jobs in batches
        for job in tqdm(jobs, desc="Processing jobs", unit="job"):
            future = executor.submit(worker, job)
            futures.append(future)
            if len(futures) >= max_workers:  # Adjust batch size as needed
                # Process completed futures before submitting more
                for future in as_completed(futures):
                    results.append(future.result())
                del futures  # Remove the reference to the future
                futures = []  # Reset the list of futures
                gc.collect()  

        # Final batch processing
        for future in as_completed(futures):
            results.append(future.result())

        return results

# @profile(stream=open("memory_profile.log", "a+"))
def create_sub_trees(file_path):
    # global_min = {'X': np.float64(2436125.219), 'Y': np.float64(229291.182), 'Z': np.float64(2381.97)}
    # global_max = {'X': np.float64(2436475.802), 'Y': np.float64(231700.191), 'Z': np.float64(2484.133)}
    boundaries = _calculate_boundaries(file_path, config['partitioning_columns'])
    input_file_basename = os.path.basename(config['input_file']).split(".")[0]
    jobs = []
    final_boundaries = {
        input_file_basename : boundaries
    }
    partitions_to_split = [config['input_file']]


    while partitions_to_split != []:
        # print("Number of partitions to split:", len(partitions_to_split))

        jobs = []
        num_paritions = len(config['partitioning_columns']) ** 3

        with Manager() as manager:
            locks = generate_locks_from_partitions(manager, partitions_to_split, num_paritions)

            for partition in partitions_to_split:
                file_size = os.path.getsize(partition)
                num_jobs = int(file_size / config['max_job_size']) + 1
                file_length = exact_line_count(partition)
                chunks = chunk_file(num_jobs, file_length)
                child_boundaries = _calculate_child_boundaries(
                    final_boundaries[os.path.basename(os.path.basename(partition)[:-4])], 
                    config["partitioning_columns"])
                boundaries = compute_partition_boundaries(partition, child_boundaries, config['partitioning_columns'])
                final_boundaries.update(boundaries)
                
                for i in range(num_jobs):
                    jobs.append((partition, chunks[i], config['partitioning_columns'], locks, child_boundaries))

            results = []
                
            manage_jobs(jobs, config['max_workers'])

            

        # Clean up old partitions
        for partition in partitions_to_split:
            if partition != config['input_file']:
                os.remove(partition)

        # Add partitions that are greater than config["max_job_size"]
        partitions_to_split = []
        for partition in os.listdir(config['tree_file']):
            if partition != os.path.basename(config['input_file']):
                partition_path = os.path.join(config['tree_file'], partition)
                partition_size = os.path.getsize(partition_path)
                if partition_size > config['max_job_size']:
                    partitions_to_split.append(partition_path)

        # If no partitions to split but more workers availible, continue splitting
        if partitions_to_split == []:
            if config["max_workers"] > len(os.listdir(config['tree_file'])):
                partitions_to_split = [os.path.join(config['tree_file'], partition) for partition in os.listdir(config['tree_file']) if partition != os.path.basename(config['input_file'])]
                

    try:
        write_pickle_dict(final_boundaries, os.path.join(config["output_dir"], "final_boundaries.pkl"))
    except Exception as e:
        print(f"Error writing final boundaries to file: {e}")
    return final_boundaries




def stage1(input_file):
    print("Starting Stage 1 ...\n_____________________________________________________________")
    start_time = time.time()
    
    config['input_file'] = input_file
    create_sub_trees(config['input_file'])

    

    # with open(log_file_path, 'r') as log_file:
    #     log_lines = log_file.readlines()
    #     peak_memory_usage = max(float(line.split()[1]) for line in log_lines if 'MiB' in line)

    print("Ending Stage 1 ...\n_____________________________________________________________")
    return {
        'file': input_file,
        'execution_time': time.time() - start_time,
        # 'peak_memory_usage': peak_memory_usage
    }


