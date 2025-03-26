import os
import laspy
import numpy as np
import pandas as pd
import pickle
from SHIELD_pipeline import shield_pipeline
from utils import las_to_csv
from collections import defaultdict
import networkx as nx
from Bio import Phylo
from visualize_trees import generate_itol_trees



def tree_edit_distance_by_level(tree1, tree2, insert_cost=1, delete_cost=1, update_cost=1):
    """
    Calculate tree edit distance at each level of the tree.
    
    Args:
        tree1: First dictionary tree
        tree2: Second dictionary tree
        insert_cost: Cost of inserting a node (default: 1)
        delete_cost: Cost of deleting a node (default: 1)
        update_cost: Cost of updating a node (default: 1)
        
    Returns:
        A dictionary mapping level numbers to edit distances
    """
    # Helper function to extract nodes at each level
    def extract_level_nodes(tree):
        """Extract nodes at each level of the tree"""
        level_nodes = defaultdict(dict)
        
        def traverse(node, path=None, level=0):
            if path is None:
                path = []
            
            # Store the current node at its level
            path_key = ".".join(path) if path else "root"
            if isinstance(node, dict):
                level_nodes[level][path_key] = {"type": "dict", "keys": set(node.keys())}
                
                # Traverse children
                for key, value in node.items():
                    traverse(value, path + [str(key)], level + 1)
            else:
                level_nodes[level][path_key] = {"type": "leaf", "value": node}
                
        traverse(tree)
        return level_nodes
    
    # Extract nodes at each level for both trees
    level_nodes1 = extract_level_nodes(tree1)
    level_nodes2 = extract_level_nodes(tree2)
    
    # Determine all levels present in either tree
    all_levels = set(level_nodes1.keys()) | set(level_nodes2.keys())
    max_level = max(all_levels) if all_levels else 0
    
    # Calculate edit distance at each level
    level_distances = {}
    level_operations = {}
    
    for level in range(max_level + 1):
        nodes1 = level_nodes1.get(level, {})
        nodes2 = level_nodes2.get(level, {})
        
        # Track operations at this level
        operations = []
        distance = 0
        
        # Find nodes only in tree1 (deletions)
        for path, node in nodes1.items():
            if path not in nodes2:
                operations.append({
                    "operation": "delete",
                    "path": path,
                    "details": node
                })
                distance += delete_cost
        
        # Find nodes only in tree2 (insertions)
        for path, node in nodes2.items():
            if path not in nodes1:
                operations.append({
                    "operation": "insert",
                    "path": path,
                    "details": node
                })
                distance += insert_cost
        
        # Find nodes in both trees that differ (updates)
        for path in set(nodes1.keys()) & set(nodes2.keys()):
            node1 = nodes1[path]
            node2 = nodes2[path]
            
            # Compare nodes
            if node1["type"] != node2["type"]:
                # Different types
                operations.append({
                    "operation": "update_type",
                    "path": path,
                    "from": node1,
                    "to": node2
                })
                distance += update_cost
            elif node1["type"] == "dict" and node2["type"] == "dict":
                # Both are dictionaries, compare keys
                keys1 = node1["keys"]
                keys2 = node2["keys"]
                
                # Calculate Levenshtein distance for key sets
                # This is simplified by counting differences
                key_differences = len(keys1 ^ keys2)  # Symmetric difference
                
                if key_differences > 0:
                    operations.append({
                        "operation": "update_keys",
                        "path": path,
                        "from_keys": keys1,
                        "to_keys": keys2,
                        "differences": key_differences
                    })
                    distance += min(key_differences * update_cost, 
                                   len(keys1 - keys2) * delete_cost + len(keys2 - keys1) * insert_cost)
            elif node1["type"] == "leaf" and node2["type"] == "leaf":
                # Both are leaves, compare values
                if node1["value"] != node2["value"]:
                    operations.append({
                        "operation": "update_value",
                        "path": path,
                        "from": node1["value"],
                        "to": node2["value"]
                    })
                    distance += update_cost
        
        level_distances[level] = distance
        level_operations[level] = operations
    
    return level_distances, level_operations

def tree_edit_distance(tree1, tree2, insert_cost=1, delete_cost=1, update_cost=1):
    """
    Calculate the tree edit distance between two dictionary trees.
    
    Args:
        tree1: First dictionary tree
        tree2: Second dictionary tree
        insert_cost: Cost of inserting a node (default: 1)
        delete_cost: Cost of deleting a node (default: 1)
        update_cost: Cost of updating a node (default: 1)
        
    Returns:
        The minimum edit distance (integer)
    """
    # Convert dictionary trees to a list-based representation with paths
    def flatten_tree(tree):
        """Convert dictionary tree to flat representation with paths"""
        flat_tree = []
        
        def traverse(node, path=None):
            if path is None:
                path = []
                
            if isinstance(node, dict):
                # Add current node
                flat_tree.append((path, "dict", len(node)))
                
                # Traverse children in sorted order for consistent results
                for key in sorted(node.keys()):
                    child_path = path + [key]
                    traverse(node[key], child_path)
            else:
                # Leaf node
                flat_tree.append((path, type(node).__name__, node))
                
        traverse(tree)
        return flat_tree
    
    # Flatten both trees
    flat_tree1 = flatten_tree(tree1)
    flat_tree2 = flatten_tree(tree2)
    
    # Initialize the distance matrix
    n = len(flat_tree1)
    m = len(flat_tree2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    # Base cases: transforming to/from empty tree
    for i in range(n + 1):
        dp[i, 0] = i * delete_cost
    for j in range(m + 1):
        dp[0, j] = j * insert_cost
    
    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            path1, type1, value1 = flat_tree1[i-1]
            path2, type2, value2 = flat_tree2[j-1]
            
            # Calculate cost of updating
            if type1 == type2:
                if type1 == "dict":
                    # For dictionaries, consider the number of keys
                    update = 0 if value1 == value2 else update_cost
                else:
                    # For leaf nodes, compare values
                    update = 0 if value1 == value2 else update_cost
            else:
                # Different types always have update cost
                update = update_cost
            
            # Tree structure also matters
            if len(path1) != len(path2):
                # Different depths always have update cost
                update = update_cost
            else:
                # Compare paths
                for p1, p2 in zip(path1, path2):
                    if p1 != p2:
                        update = update_cost
                        break
            
            # Calculate minimum edit distance
            dp[i, j] = min(
                dp[i-1, j] + delete_cost,      # Delete from tree1
                dp[i, j-1] + insert_cost,      # Insert from tree2
                dp[i-1, j-1] + update          # Update or keep
            )
    
    return dp[n, m]

def label_tree_differences(tree1, tree2):
    """
    Label all differences between two trees.
    
    Args:
        tree1: First tree (dictionary)
        tree2: Second tree (dictionary)
        
    Returns:
        A dictionary containing labeled differences
    """
    differences = {}
    
    def compare_trees(t1, t2, path="root"):
        # Handle case where one node exists and the other doesn't
        if t1 is None and t2 is not None:
            differences[path] = {"status": "added", "value": t2}
            return
        elif t1 is not None and t2 is None:
            differences[path] = {"status": "removed", "value": t1}
            return
        
        # Both are None
        if t1 is None and t2 is None:
            return
            
        # Compare values for leaf nodes or node types
        if not isinstance(t1, dict) or not isinstance(t2, dict):
            if t1 != t2:
                differences[path] = {
                    "status": "changed",
                    "old_value": t1,
                    "new_value": t2
                }
            return
        
        # Both are dictionaries - compare keys
        all_keys = set(t1.keys()) | set(t2.keys())
        
        for key in all_keys:
            new_path = f"{path}.{key}" if path != "root" else key
            
            if key not in t1:
                differences[new_path] = {"status": "added", "value": t2[key]}
            elif key not in t2:
                differences[new_path] = {"status": "removed", "value": t1[key]}
            else:
                # Recursively compare the values
                compare_trees(t1[key], t2[key], new_path)
    
    compare_trees(tree1, tree2)
    return differences


def count_nodes_by_level(dictionary):
    """
    Count the number of nodes at each level in a nested dictionary.
    
    Args:
        dictionary: A nested dictionary
        
    Returns:
        A dictionary where keys are level numbers (starting from 0)
        and values are the number of nodes at that level
    """
    level_counts = {}
    
    def traverse(current_dict, level=0):
        # Ensure this level exists in our counts
        if level not in level_counts:
            level_counts[level] = 0
            
        # Count the current node
        level_counts[level] += 1
        
        # Recursively traverse each child that is a dictionary
        for key, value in current_dict.items():
            if isinstance(value, dict):
                traverse(value, level + 1)
    
    # Start the traversal
    traverse(dictionary)
    return level_counts


def networkx_to_newick(graph, root_node=None):
    """
    Convert a NetworkX DiGraph to Newick format.
    
    Args:
        graph: NetworkX DiGraph representing a tree
        root_node: The root node of the tree (will be determined automatically if None)
        
    Returns:
        A string in Newick format
    """
    if root_node is None:
        # Find a root (node with no predecessors)
        for node in graph.nodes():
            if graph.in_degree(node) == 0:
                root_node = node
                break
    
    def _get_newick(node, parent=None):
        children = [child for child in graph.successors(node) if child != parent]
        
        if not children:
            # For leaf nodes, use the label if available
            label = graph.nodes[node].get('label', str(node))
            return f"{label}"
        
        # Recursively handle children
        subtrees = [_get_newick(child, node) for child in children]
        label = graph.nodes[node].get('label', str(node))
        
        return f"({','.join(subtrees)}){label}"
    
    newick_str = _get_newick(root_node) + ";"
    return newick_str


def dict_to_networkx(dictionary, parent=None, graph=None, node_id=0):
    """
    Convert a dictionary-based tree to a NetworkX graph.
    
    Args:
        dictionary: A dictionary representing a tree
        parent: Parent node ID (for edge creation)
        graph: Existing NetworkX graph (created if None)
        node_id: Current node ID counter
        
    Returns:
        A tuple of (NetworkX graph, next node ID)
    """
    if graph is None:
        graph = nx.DiGraph()
    
    current_id = node_id
    node_id += 1
    
    # For the root node, use the first key if available
    if parent is None:
        if isinstance(dictionary, dict):
            # Create a root node
            root_label = "root"
            graph.add_node(current_id, label=root_label)
            
            # Process children
            for key, value in dictionary.items():
                next_id, node_id = dict_to_networkx(value, current_id, graph, node_id)
                # Add edge with key as label
                graph.add_edge(current_id, next_id, label=key)
        else:
            # Single value as root
            graph.add_node(current_id, label=str(dictionary))
    else:
        if isinstance(dictionary, dict):
            # Create intermediary node
            graph.add_node(current_id, label="")
            
            # Create edges to children
            for key, value in dictionary.items():
                next_id, node_id = dict_to_networkx(value, current_id, graph, node_id)
                graph.add_edge(current_id, next_id, label=key)
        else:
            # Leaf node
            graph.add_node(current_id, label=str(dictionary))
            
        # Connect to parent
        if parent is not None:
            graph.add_edge(parent, current_id)
    
    return graph, node_id 

def pre_process_tree(tree, index=""):
    # remove all "data" and "boundaries" keys
    def remove_keys(node, index):
        if "data" in node:
            node.pop("data")
            node[index] = {}
        if "boundaries" in node:
            node.pop("boundaries")
    remove_keys(tree, index)
    for key, value in tree.items():
        if isinstance(value, dict):
            if index == "":
                pre_process_tree(value, str(key))
            else:
                pre_process_tree(value, str(index)+"_"+str(key))

def generate_tree(output_dir):
    """
    Function to generate a tree from the pkl file
    """
    def get_index_from_string(string):
        index = string.split("_")[-1][:-4]
        return index.split("-")



    root = {}
    # Iterate through all the files in the output directory
    for filename in os.listdir(output_dir):
        if filename.endswith(".pkl"):
            with open(os.path.join(output_dir, filename), "rb") as f:
                # Load the pkl file
                tree_data = pickle.load(f)
                index = get_index_from_string(filename)
                # Create a nested dictionary structure
                current_level = root
                for i in index[:-1]:
                    if i not in current_level:
                        current_level[i] = {}
                    current_level = current_level[i]
                # Add the data to the leaf node
                if current_level.get(index[-1]) is not None:
                    raise ValueError(f"Duplicate key found: {index[-1]}")
                current_level[index[-1]] = tree_data
    
    pre_process_tree(root)

    return root



def apply_transformation_sphere(spheres, scale, offset, validation_las):
    new_spheres = spheres
    # new_spheres = []
    # for sphere in spheres:
    #     # Apply transformation to the sphere's center
    #     sphere_center = sphere[1] * scale + offset
    #     sphere_radius = sphere[0] * scale
    #     new_spheres.append((sphere_radius, sphere_center))

    # Check if the transformed spheres are within the bounds of the LAS file
    las_min = np.array([validation_las.header.x_min, validation_las.header.y_min, validation_las.header.z_min])
    las_max = np.array([validation_las.header.x_max, validation_las.header.y_max, validation_las.header.z_max])

    for sphere in new_spheres:
        sphere_radius, sphere_center = sphere
        if (sphere_center[0] - sphere_radius < las_min[0] or sphere_center[0] + sphere_radius > las_max[0] or
            sphere_center[1] - sphere_radius < las_min[1] or sphere_center[1] + sphere_radius > las_max[1]):
            raise ValueError("Transformed sphere is out of bounds.")

    return new_spheres

def apply_transformation_box(bounding_box, scale, offset, validation_las):
    transformed_bounding_box = bounding_box
    # transformed_bounding_box = (
    #     (bounding_box[0][0] * scale[0] + offset[0], bounding_box[0][1] * scale[0] + offset[0]),
    #     (bounding_box[1][0] * scale[1] + offset[1], bounding_box[1][1] * scale[1] + offset[1]),
    #     (bounding_box[2][0] * scale[2] + offset[2], bounding_box[2][1] * scale[2] + offset[2])
    # )

    # Check if the transformed bounding box is within the bounds of the LAS file
    las_min = np.array([validation_las.header.x_min, validation_las.header.y_min, validation_las.header.z_min])
    las_max = np.array([validation_las.header.x_max, validation_las.header.y_max, validation_las.header.z_max])

    if las_min[0] > transformed_bounding_box[0][0] or las_max[0] < transformed_bounding_box[0][1]:
        raise ValueError("Transformed bounding box is out of bounds in the X direction.")
    if las_min[1] > transformed_bounding_box[1][0] or las_max[1] < transformed_bounding_box[1][1]:
        raise ValueError("Transformed bounding box is out of bounds in the Y direction.")


    return transformed_bounding_box
    

def convert_laz_to_las(in_laz, out_las):
    las = laspy.read(in_laz)
    las = laspy.convert(las)
    las.write(out_las)        

def sub_select_bounding_box(las, bounding_box):
    """
    Sub-selects points within a given bounding box from a LAS object.
    
    Parameters:
        las (laspy.LasData): A laspy LAS object.
        bounding_box (tuple): ((xmin, xmax), (ymin, ymax), (zmin, zmax)) defining the bounding box.
        
    Returns:
        laspy.LasData: Sub-selected LAS object.
    """
    x_min, x_max = bounding_box[0]
    y_min, y_max = bounding_box[1]
    z_min, z_max = bounding_box[2]

    mask = (
        (las.x >= x_min) & (las.x <= x_max) &
        (las.y >= y_min) & (las.y <= y_max) &
        (las.z >= z_min) & (las.z <= z_max)
    )

    new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    new_las.points = las.points[mask]

    return new_las


def carve_hole(las, center, radius, removal_fraction=0.5):
    """
    Removes a random fraction of points within a given sphere from a LAS object.
    
    Parameters:
        las (laspy.LasData): A laspy LAS object.
        center (tuple): (x, y, z) coordinates of the sphere's center.
        radius (tuple): (rx, ry, rz) defining the sphere's radii along each axis.
        removal_fraction (float): Fraction of points to randomly remove within the sphere (0 to 1).
        
    Returns:
        laspy.LasData: Modified LAS object with some points randomly removed inside the sphere.
    """
    x, y, z = center
    rx, ry, rz = radius

    # Compute squared distances to avoid sqrt for efficiency
    distances_sq = ((las.x - x) / rx) ** 2 + ((las.y - y) / ry) ** 2 + ((las.z - z) / rz) ** 2

    # Identify points inside the ellipsoidal region
    inside_sphere = distances_sq <= 1

    # Get indices of points inside the sphere
    inside_indices = np.where(inside_sphere)[0]

    # Randomly select a subset of these points to remove
    num_to_remove = int(len(inside_indices) * removal_fraction)
    if num_to_remove > 0:
        removal_indices = np.random.choice(inside_indices, num_to_remove, replace=False)
    else:
        removal_indices = np.array([])

    # Create a mask to keep only points not in removal_indices
    mask = np.ones(len(las.x), dtype=bool)
    mask[removal_indices] = False

    # Apply the mask to filter the points
    new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    new_las.points = las.points[mask]

    return new_las


def generate_damages(las_file, bounding_box, spheres, output_file, undamaged_las_file):
    """
    Function to generate damages in the LAS file based on the provided bounding box and spheres.
    
    Parameters:
        las_file (str): Path to the input LAS file.
        bounding_box (tuple): ((xmin, xmax), (ymin, ymax), (zmin, zmax)) defining the bounding box.
        spheres (list): List of tuples containing center and radius for each sphere.
        
    Returns:
        None
    """
    
    las = laspy.read(las_file)
    # Sub-select points within the bounding box
    # bounding_box = apply_transformation_box(bounding_box, scale=las.header.scale, offset=las.header.offset, validation_las=las)
    sub_las = sub_select_bounding_box(las, bounding_box)
    sub_las_undamaged = sub_select_bounding_box(las, bounding_box)
    
    # Carve holes in the sub-selected LAS file
    # spheres = apply_transformation_sphere(spheres, scale=las.header.scale, offset=las.header.offset, validation_las=las)
    for sphere in spheres:
        radius, center = sphere
        sub_las = carve_hole(sub_las, center, radius, removal_fraction=1)

    
    # Save the modified LAS file
    sub_las.write(output_file)
    sub_las_undamaged.write(undamaged_las_file)




def random_sample_dataset(las_file, output_file, sample_percentage):
    """
    Function to random sample the dataset for testing
    """
    
    las = laspy.read(las_file)
    num_points = len(las.points)
    num_sampled_points = int(num_points * sample_percentage)
    sampled_indices = np.random.choice(num_points, num_sampled_points, replace=False)
    sampled_points = las.points[sampled_indices]

    las.points = sampled_points
    las.write(output_file)

    print(f"Sampled LAS file saved to {output_file}")

def convert_las_to_csv(las_file, csv_file):
    """
    Function to convert LAS file to CSV
    """
    las_to_csv(las_file, csv_file)
    print(f"Converted {las_file} to {csv_file}")

def run_shield_on_datasets():
    """
    Function to run SHIELD on the datasets
    """



if __name__ == "__main__":
    # print("Starting Damage Detection Pipeline")
    # # Decompress laz file
    # laz_file = "data/lidar_data/AVST0069_A10.laz"
    # las_file = laz_file[:-4] + ".las"
    # convert_laz_to_las(laz_file, las_file)



    # Generate Damages
    las_file = "data/lidar_data/AVST0069_A10.las"
    bounding_box_center = 2436290.765, 230778.356, 2430.159
    bounding_box_size = 229.432, 193.675, 81.360
    bounding_box = (
        (bounding_box_center[0] - bounding_box_size[0] / 2, bounding_box_center[0] + bounding_box_size[0] / 2),
        (bounding_box_center[1] - bounding_box_size[1] / 2, bounding_box_center[1] + bounding_box_size[1] / 2),
        (bounding_box_center[2] - bounding_box_size[2] / 2, bounding_box_center[2] + bounding_box_size[2] / 2)
    )
    spheres = [
            ((12.736, 12.736, 12.736), (2436364.146, 230853.643, 2432.263))
        ]

    damaged_las_file = "data/lidar_data/AVST0069_A10_damaged_0.las"
    undamaged_las_file = "data/lidar_data/AVST0069_A10_undamaged_0.las"
    generate_damages(las_file, bounding_box, spheres, damaged_las_file, undamaged_las_file)
    
    


    # Random sample dataset twice for testing
    sample_percentage = 0.1  # 10% of the dataset

    random_sample_dataset(undamaged_las_file, undamaged_las_file, sample_percentage)
    random_sample_dataset(damaged_las_file, damaged_las_file, sample_percentage)


    # Convert las files to csv
    undamaged_csv = undamaged_las_file.replace('.las', '.csv')
    damaged_csv = damaged_las_file.replace('.las', '.csv')
    convert_las_to_csv(undamaged_las_file, undamaged_csv)
    convert_las_to_csv(damaged_las_file, damaged_csv)


    # Run SHIELD on both datasets
    undamaged_test = {
        "file": undamaged_csv,
        "scale": 0.1
    }
    damaged_test = {
        "file": damaged_csv,
        "scale": 0.1
    }
    shield_pipeline(undamaged_test)
    shield_pipeline(damaged_test)


    # Compare trees and generate report
    undamaged_tree = generate_tree("reduced results/AVST0069_A10_undamaged_0_0.1/tree_partitions/")
    damaged_tree = generate_tree("reduced results/AVST0069_A10_damaged_0_0.1/tree_partitions/")
    # graph1, _ = dict_to_networkx(undamaged_tree)
    # graph2, _ = dict_to_networkx(damaged_tree)

        # Fix: Use a custom stringizer function
    def my_stringizer(value):
        if isinstance(value, (str, int, float, bool)):
            return value
        else:
            return str(value)

    # nx.write_gml(graph1, "tree1.gml", stringizer=my_stringizer)
    # nx.write_gml(graph2, "tree2.gml", stringizer=my_stringizer)

    
    # newick_string = networkx_to_newick(graph1)
    # newick_string2 = networkx_to_newick(graph2)

    # # Save to file
    # with open("tree.newick", "w") as f:
    #     f.write(newick_string)

    # with open("tree2.newick", "w") as f:
    #     f.write(newick_string2)

    print(count_nodes_by_level(undamaged_tree))
    print(count_nodes_by_level(damaged_tree))
    # print(tree_edit_distance(undamaged_tree, damaged_tree))
    print(tree_edit_distance_by_level(undamaged_tree, damaged_tree)[0])
    

    generate_itol_trees(undamaged_tree, damaged_tree)