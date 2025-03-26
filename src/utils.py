import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import laspy
from pathlib import Path
import shutil
import h5py
import glob

with open('config.json', 'r') as file:
    config = json.load(file)
    
def safe_write_to_csv(file_path, data, lock):
    with lock:
        try:
            write_header = not os.path.exists(file_path)
            data.to_csv(file_path, mode='a', header=write_header, index=False)
        except Exception as e:
            print(f"Failed to write to {file_path}: {e}")

def unsafe_write_to_csv(data, file_path):
    try:
        write_header = not os.path.exists(file_path)
        data.to_csv(file_path, mode='a', header=write_header, index=False)
    except Exception as e:
        print(f"Failed to write to {file_path}: {e}")

def reset_directory(target_directory):
    # List all items in the target_directory
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)
    for item in os.listdir(target_directory):
        item_path = os.path.join(target_directory, item)
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Recursively remove the directory and its contents
            shutil.rmtree(item_path)
        else:
            # Remove the file
            os.remove(item_path)

    # Add folder tree_partitions folder
    os.makedirs(os.path.join(target_directory, 'tree_partitions'), exist_ok=True)
    os.makedirs(os.path.join(target_directory, 'vectors'), exist_ok=True)


def analyze_results(results):
    sizes = [result['size_percentage'] for result in results]
    times = [result['execution_time'] for result in results]
    memory_usages = [result['peak_memory_usage'] for result in results]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sizes, times, marker='o')
    plt.xlabel('Dataset Size (%)')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Dataset Size')

    plt.subplot(1, 2, 2)
    plt.plot(sizes, memory_usages, marker='o')
    plt.xlabel('Dataset Size (%)')
    plt.ylabel('Peak Memory Usage (MiB)')
    plt.title('Memory Usage vs Dataset Size')

    plt.tight_layout()
    plt.savefig('analysis_results.png')
    plt.show()


def get_loss(importance, stds, dimensions=3, scale=100, reduction_columns=None):
    '''
    Calculate the maximum allowed loss for each variable based on the importance, standard deviation and user-defined "scale" parameter.
    
    Parameters:
    importance (dict): Dictionary containing the importance of each variable
    std (dict): Dictionary containing the standard deviation of each variable
    dimensions (int): Number of dimensions to consider
    scale (int): User-defined parameter to scale the loss
    
    Returns:
    dict: Dictionary containing the maximum allowed loss for each variable
    '''
    try:
        loss = {}
        for key in reduction_columns:
            if importance[key] < 0.02: # Ignore variables with importance less than 0.02
                continue
            loss_value = stds[key] * scale
            loss[key] = (loss_value , 0)
            if len(loss) == dimensions:
                break
    except Exception as e:
        print(f"Error in get_loss: {e}")

    return loss


def extract_data(dictionary):
    """Recursively extract mean data from a nested dictionary."""
    if "data" in dictionary:
        return [pd.DataFrame(dictionary["data"]).mean().to_frame().T]
    
    return [
        mean_data
        for key, value in dictionary.items()
        if key != "boundaries"
        for mean_data in extract_data(value)
    ]

    
def read_pkl(file):
    # for each pkl file, read data and append to a list
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def conglomerate_pkl_files(dir, output_file):
    # Get all .pkl files in the directory
    pkl_files = glob.glob(os.path.join(dir, "*.pkl"))

    # Read all pkl files into a list of DataFrames
    dfs = [extract_data(read_pkl(file)) for file in pkl_files]
    conglomerated_file = pd.DataFrame()
    for df_list in dfs:
        df = pd.concat(df_list, ignore_index=True)
        conglomerated_file = pd.concat([conglomerated_file, df], ignore_index=True)
    conglomerated_file.to_csv(output_file) 

def conglomerate_csv_files(directory, output_file):
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    if not csv_files:
        print("No CSV files found in the directory.")
        return
    
    # Read all CSVs and store as list of DataFrames
    dfs = [pd.read_csv(file) for file in csv_files]

    # Concatenate all DataFrames
    final_df = pd.concat(dfs, ignore_index=True)

    # Write to output file
    final_df.to_csv(output_file, index=False)



def pandas_to_las(df, old_las, output_file):
    point_format = old_las.header.point_format.id
    version = str(old_las.header.version.major) + "." + str(old_las.header.version.minor)
    header = laspy.LasHeader(point_format=point_format, version=version)
    new_las = laspy.LasData(header)

    for column in df.columns:
        new_las.points[column] = df[column].astype(type(old_las[column][0]))

    new_las.write(output_file)

def count_lines(file):
    with open(file) as f:
        return sum(1 for line in f)

def compare_csv_files(file1, file2):
    # Compare the lengths of the two files
    length1 = count_lines(file1)
    length2 = count_lines(file2)
    return length2 / length1


def generate_vectors(possible_vectors, populated_nodes, parent_node, means, stds, leaf_status):
    """
    Generate vectors for the given populated nodes based on the possible vectors.
    
    Args:
        possible_vectors (list): List of possible vectors.
        populated_nodes (list): List of populated nodes.
        
    Returns:
        list: List of generated vectors.
    """
    threshold = 0.25
    angle_threshold = 30



    # Filter out line combinations that contain indices that are not in the parent_dict
    def filter_combinations(nodes, combinations):
        populated_set = set(populated_nodes)
        line_combos_filtered = [combo for combo in possible_vectors if all(node in populated_set for node in combo)]
        return line_combos_filtered

    
    # Find nodes that contain at least 1 point and generate a list of valid lines between them.
    line_combos_filtered = filter_combinations(populated_nodes, possible_vectors)
    try:
        validated_vectors = []
        for vector in line_combos_filtered:
            # check if the mid point is acceptable:
            ratio_distance = np.linalg.norm(np.array(means[vector[0]]) - np.array(means[vector[1]])) / np.linalg.norm(np.array(means[vector[1]]) - np.array(means[vector[2]]))
            if abs(ratio_distance - 1) < threshold: # check ratio distance
                if test_valid_vector((means[vector[0]], means[vector[1]], means[vector[2]]), angle_threshold):
                    vector_leaf_status = [leaf_status[vector[0]], leaf_status[vector[1]], leaf_status[vector[2]]]
                    validated_vectors.append([means[vector[0]], means[vector[2]], vector_leaf_status, parent_node])
    except:
        raise ValueError("Error validating vector varience or error angle.")

    return pd.DataFrame(validated_vectors, columns=['P1', "P2", 'leaf_status', 'depth'])


import h5py
import numpy as np
from multiprocessing import Lock

def append_np_array(np_array, filename, dataset_name):
    if np_array is None or np_array.size == 0:
        # print("[INFO] Empty array detected, skipping append.")
        return
    try:
        if not os.path.exists(filename):
            with h5py.File(filename, 'w') as f:
                maxshape = (None,) + np_array.shape[1:]  # Allow unlimited growth in first dimension
                f.create_dataset(dataset_name, data=np_array, maxshape=maxshape, chunks=True)
        else:
            with h5py.File(filename, 'a') as f:
                # print(f"[INFO] Opening file: {filename} (Attempt {attempt+1})")
                if dataset_name in f:
                    dataset = f[dataset_name]

                    if dataset.shape[1:] != np_array.shape[1:]:
                        raise ValueError(f"[ERROR] Shape mismatch! Existing: {dataset.shape[1:]}, New: {np_array.shape[1:]}")

                    original_shape = dataset.shape
                    new_shape = (original_shape[0] + np_array.shape[0],) + original_shape[1:]

                    dataset.resize(new_shape)  # Resize dataset
                    dataset[-np_array.shape[0]:] = np_array  # Append new data
                else:
                    maxshape = (None,) + np_array.shape[1:]  # Allow unlimited growth in first dimension

                    f.create_dataset(dataset_name, data=np_array, maxshape=maxshape, chunks=True)

    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")






def test_valid_vector(vector, tolerance_angle):
    # visualize_test_valid_vector(vector, tolerance_angle)
    P1, P2, P3 = vector
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)
    # checks angle of error in vector is within tolerance

    # Compute vectors P1P2 and P1P3
    vec_P1P2 = P2 - P1
    vec_P1P3 = P3 - P1
    vec_P2P3 = P3 - P2

    # Compute the dot product
    dot_productP2P1P3 = np.dot(vec_P1P2, vec_P1P3)
    dot_productP1P2P3 = np.dot(vec_P1P2, vec_P2P3)

    # Compute the magnitudes
    magnitude_P1P2 = np.linalg.norm(vec_P1P2)
    magnitude_P1P3 = np.linalg.norm(vec_P1P3)
    magnitude_P2P3 = np.linalg.norm(vec_P2P3)

    # Compute the cosine of the angle
    cos_thetaP2P1P3 = dot_productP2P1P3 / (magnitude_P1P2 * magnitude_P1P3)
    cos_thetaP1P2P3 = dot_productP1P2P3 / (magnitude_P1P2 * magnitude_P2P3)


    # Compute the angle in radians
    theta_radiansP2P1P3 = np.arccos(cos_thetaP2P1P3)
    theta_radiansP1P2P3 = np.arccos(cos_thetaP1P2P3)

    # Convert to degrees if needed
    theta_degreesP2P1P3 = np.degrees(theta_radiansP2P1P3)
    P2P1P3 = theta_degreesP2P1P3 < tolerance_angle or theta_degreesP2P1P3 > 360 - tolerance_angle
    theta_degreesP1P2P3 = np.degrees(theta_radiansP1P2P3)
    P1P2P3 = theta_degreesP1P2P3 < tolerance_angle or theta_degreesP1P2P3 > 360 - tolerance_angle

    return P2P1P3 and P1P2P3
    

def visualize_test_valid_vector(vector, tolerance_angle):
    P1, P2, P3 = vector
    P1 = np.array(P1)
    P2 = np.array(P2)
    P3 = np.array(P3)

    """
    Plots a line between P1 and P3 and displays P2 on the graph.
    
    Parameters:
    P1, P2, P3: Tuples representing (x, y) coordinates of the points.
    """
    # Extracting coordinates
    x_values = [P1[0], P3[0]]
    y_values = [P1[1], P3[1]]
    
    # Creating the plot
    plt.figure(figsize=(6,6))
    plt.plot(x_values, y_values, 'b-', label='Line from P1 to P3')  # Line from P1 to P3
    plt.scatter([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]], color=['red', 'green', 'blue'])  # Plot points
    
    # Annotate the points
    plt.text(P1[0], P1[1], ' P1', verticalalignment='bottom', fontsize=12, color='red')
    plt.text(P2[0], P2[1], ' P2', verticalalignment='bottom', fontsize=12, color='green')
    plt.text(P3[0], P3[1], ' P3', verticalalignment='bottom', fontsize=12, color='blue')
    
    # Formatting
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Line Between P1 and P3 with P2 Marked")
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

def las_to_csv(file_path, csv_path):
    with laspy.open(file_path) as las_file:
        las_columns = [x[0] for x in las_file.header.point_format.dimensions]
        header = pd.DataFrame(columns=las_columns)
        header.to_csv(csv_path, index=False)

        # If you want to process points in chunks:
        current_line = 0
        chunk_size = 100000
        for points in las_file.chunk_iterator(chunk_size):
            chunk = pd.DataFrame(points.array, columns=las_columns)
            current_line += len(chunk)
            print("Processing chunk...")
            chunk.to_csv(csv_path, mode="a", header=False, index=False)

            print(f"{100 * current_line / las_file.header.point_count} % processed.")