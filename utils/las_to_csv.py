import laspy
import pandas as pd
import csv


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