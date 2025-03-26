import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Lock, current_process
import json
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from utils import get_loss, generate_vectors, safe_write_to_csv, append_np_array, unsafe_write_to_csv

def write_pickle_dict(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# @profile
def _partition_data(df, child_boundaries, partition_columns):
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
    # Instead of assigning partition_index to df, you can group by it using pd.groupby on a Series
    partitions = {}
    grouped = df.groupby(pd.Series(partition_index, index=df.index), sort=False)
    del df

    for grp_partition_index, subset in grouped:
        # Extract boundaries for the current partition
        bin_indices = np.unravel_index(grp_partition_index, bin_sizes)
        boundaries = {
            col: boundaries_dict[col][idx]
            for col, idx in zip(partition_columns, bin_indices)
        }

        # Store data (the subset) and boundaries in the partitions dictionary
        partitions[grp_partition_index] = {
            'data': subset,
            'boundaries': boundaries
        }

    # ensure that no points were dropped:
    # total_points = sum([len(partition['data']) for partition in partitions.values()])
    # if total_points != len(df):
    #     raise ValueError(f"Dataframe length {len(df)} does not match the total points in the partitions {total_points}")
    for partition in partitions:
        if "data" not in partitions[partition]:
            print("91 Data not in partition")
    return partitions

def _calculate_child_boundaries(parent_boundary, partition_columns):
    '''
    Calculate the boundaries of the child nodes given the boundaries of the parent node.
    
    Parameters:
    parent_boundary (list): The boundaries of the parent node.
    
    Returns:
    dict: The boundaries of the child nodes.
    '''
    child_boundaries = {}
    for i, col in enumerate(partition_columns):
        # Get the boundaries for the current column
        diff = parent_boundary[col][1] - parent_boundary[col][0]
        child_boundaries[col] = ((parent_boundary[col][0],
                                parent_boundary[col][0] + (diff / 3),
                                parent_boundary[col][0] + (diff / 3) * 2,
                                parent_boundary[col][1] + (diff / 3) * 3))
    return child_boundaries


def leaf_test(current_node, tolerance, current_node_mean):
    if "data" not in current_node:
        print("data not in current_node")
        return True
    df = current_node["data"]
    # If there is only one value in the dataframe, pass by default
    try:
        if df.shape[0] <= 1:
            return True
    except Exception as e:
        print(f"Error in leaf_test: {e}")

    # Create a copy of the dataframe to avoid changing the original
    df_copy = df.copy()

    # If df_copy is a Series, convert it to a DataFrame
    if isinstance(df_copy, pd.Series):
        df_copy = df_copy.to_frame()

    # Determine if the maximum loss is within the acceptable tolerance
    try:
        for column in tolerance.keys():
            loss = max(abs(current_node_mean[column] - df_copy[column].max()), 
                    abs(current_node_mean[column] - df_copy[column].min()))
            if loss > tolerance[column][0]:
                return False
    except Exception as e:
        print(f"Error in leaf_test (loss): {e}")
        print(loss)
        print(tolerance[column])

    return True

# @profile
def compute_sub_tree(sub_tree_path, current_node, tolerance, depth, vector_lock, current_node_mean, config):
    # check if the current node can be a leaf node
    
    if leaf_test(current_node, tolerance, current_node_mean):
        mean_point = current_node_mean.to_frame().T
        mean_point.reset_index(drop=True, inplace=True)
        return current_node, mean_point, 1
    else:
        # create the children of the current node
        try:
            child_boundaries = _calculate_child_boundaries(current_node["boundaries"], config['partitioning_columns'])
            partitions = _partition_data(current_node["data"], child_boundaries, config['partitioning_columns'])
        except Exception as e:
            raise ValueError(f"Error in partitioning data: {e}")
        # remove the data from the current node
        current_node.pop("data")

        partitions_means = {i: data["data"].mean()[config["reduction_columns"]] for i, data in partitions.items()}
        
        partitions_stds = {i: data["data"].std()[config["reduction_columns"]] for i, data in partitions.items()}

        df_result = pd.DataFrame()
        leaf_status = {}
        for index, partition in partitions.items():
            # add the partition to the current node
            current_node[index], df, leaf = compute_sub_tree(sub_tree_path, partition, tolerance, depth + "-" + str(index), vector_lock, partitions_means[index], config)
            leaf_status[index] = leaf
            df_result = pd.concat([df_result, df], ignore_index=True)
        
        
        # Generate Vectors
        try:
            vectors = generate_vectors(config["vector_set"], list(partitions.keys()), depth, partitions_means, partitions_stds, leaf_status)
            # Write vectors to csv
            unsafe_write_to_csv(vectors, config["output_dir"] + "/vectors/" + os.path.basename(sub_tree_path[:-4] + "_vectors.csv"))
        except Exception as e:
            raise ValueError(f"Error in generating vectors for {sub_tree_path}, Error: {e}")

        return current_node, df_result, 0
    


# @profile
def recursive_worker(args):
    sub_tree_path, boundaries, tolerance, vector_lock, config = args
    try:
        current_node = {"data": pd.read_csv(sub_tree_path), "boundaries": boundaries}
    except Exception as e:
        print(f"Error reading csv file{sub_tree_path} (recursive): {e}")
    sub_tree_int = "-".join(sub_tree_path.split("-")[1:]).split(".")[0]
    sub_tree, df_result, _ = compute_sub_tree(sub_tree_path, current_node, tolerance, sub_tree_int, vector_lock, current_node["data"].mean(), config)
    # save results
    os.remove(sub_tree_path)
    df_result.to_csv(sub_tree_path)
    sub_tree_path = sub_tree_path.split(".")[0] + ".pkl"
    write_pickle_dict(sub_tree, sub_tree_path)
    del sub_tree
    del df_result

with open('config.json', 'r') as file:
    config = json.load(file)


def stage2(input_file, scale):
    global config
    with open('config.json', 'r') as file:
        config = json.load(file)
    dims = len(config["partitioning_columns"])
    with open(f"data/line_combos/line_combos_{dims}d.pkl", mode="rb") as f:
        config["vector_set"] = pickle.load(f)
    config["scale"] = scale
    start_time = time.time()
    final_boundaries = pickle.load(open(config['output_dir'] + '/' + "final_boundaries.pkl", 'rb'))

    if "pca_results" in config:
        pca_results = pd.read_csv(config['pca_results'])
        importance = pca_results[["feature_name", "relative_importances"]].set_index("feature_name").to_dict()["relative_importances"]
        stds = pca_results[["feature_name", "overall_std"]].set_index("feature_name").to_dict()["overall_std"]
        tolerance = get_loss(importance, stds, dimensions=3, scale=config["scale"], reduction_columns=config['reduction_columns'])
    elif "tolerance" in config:
        tolerance = config["tolerance"]
    else:
        raise ValueError("No tolerance or pca_results found in config file")
    

    results = []
    with Manager() as manager:
        with ProcessPoolExecutor(max_workers=config["max_workers"]) as executor:
            futures = []
            for sub_tree in os.listdir(config['tree_file']):
                boundary = final_boundaries[sub_tree.split(".")[0]]
                vector_lock = manager.Lock()
                args = (os.path.join(config['tree_file'], sub_tree), boundary, tolerance, vector_lock, config)
                futures.append(executor.submit(recursive_worker, args))

            # Collect results and handle errors
            for future in tqdm(futures, total=len(futures), desc="Processing subtrees", unit="subtree"):
                try:
                    results.append(future.result())  # Fetch worker results
                except Exception as e:
                    print(f"Error in worker {future}: {e}")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

# if __name__ == "__main__":
#     stage2()
