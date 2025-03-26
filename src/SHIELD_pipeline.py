import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import laspy
import sys
import json
from memory_profiler import LogFile
import time
import shutil

from utils import reset_directory, conglomerate_csv_files, pandas_to_las, compare_csv_files, count_lines
from Stage1 import stage1
from Stage2 import stage2


# log_file_path = "memory_profile.log"
# log_file = open(log_file_path, "w+")
# sys.stdout = LogFile(log_file_path, reportIncrementFlag=False)

def csv_to_las(file_path):
    # Open the .laz file
    csv_path = file_path[:-4] + ".csv"
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

def calculate_size_ratio(original_file, new_file):
    # original_file_length = count_lines(original_file)

    data_reduction = compare_csv_files(original_file, new_file)
    percent_reduced = data_reduction*1000
    # print(f"{dataset['file']} is {percent_reduced/10}% of the original size")
    return percent_reduced



def shield_pipeline(dataset):
    with open("config.json") as f:
        config = json.load(f)
    test_name = os.path.basename(dataset["file"][:-4]) + "_" + str(dataset["scale"])
    print(f"Processing dataset: {dataset['file']} at scale = ({dataset['scale']})")
    total_time = time.time()
    config['scale'] = dataset['scale']
    config['input_file'] = dataset['file']

    reset_directory(config["output_dir"])

    stage1_time = time.time()
    stage1(dataset["file"])
    stage1_time = time.time() - stage1_time


    stage2_time = time.time()
    stage2(config['input_file'], config['scale'])
    stage2_time = time.time() - stage2_time

    reduced_file_name = f"reduced_{dataset['file'].split('/')[-1]}"

    conglomerate_time = time.time()
    conglomerate_csv_files(config['tree_file'], f"out/{reduced_file_name}")
    conglomerate_time = time.time() - conglomerate_time

    total_time = time.time() - total_time

    percent_reduced = calculate_size_ratio(dataset["file"], f"out/{reduced_file_name}")


    advanced_results = {
        "Test Name": test_name,
        "Filename": dataset["file"],
        "scale": dataset['scale'],
        "Runtime (s)": total_time,
        "Stage 1 time (s)": stage1_time,
        "Stage 2 time (s)": stage2_time,
        "Conglomerate Time": conglomerate_time,
        "Reduction/Original Ratio": percent_reduced/1000,
    }
    if not os.path.exists("reduced results"):
        os.makedirs("reduced results")
    pd.DataFrame([advanced_results]).to_csv(f"reduced results/advanced_test_results.csv", index=False, mode='a', header=False)

    # Save Results
    results_dir = f"reduced results/{test_name}/"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    shutil.move("out", results_dir)


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)

    for dataset in config["datasets"]:
        shield_pipeline(dataset)