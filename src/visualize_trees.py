import os
import json
from collections import defaultdict

def identify_tree_differences(tree1, tree2):
    """
    Identify differences between two trees.
    
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
            differences[path] = {"status": "added"}
            return
        elif t1 is not None and t2 is None:
            differences[path] = {"status": "removed"}
            return
        
        # Both are None
        if t1 is None and t2 is None:
            return
            
        # Only focus on dictionary structure, ignore non-dict data
        if not isinstance(t1, dict) and not isinstance(t2, dict):
            # Both are leaf nodes, but we don't care about their values
            return
        elif not isinstance(t1, dict):
            # t1 is a leaf but t2 is a dictionary
            differences[path] = {"status": "structure_changed", "details": "leaf_to_branch"}
            return
        elif not isinstance(t2, dict):
            # t2 is a leaf but t1 is a dictionary
            differences[path] = {"status": "structure_changed", "details": "branch_to_leaf"}
            return
        
        # Both are dictionaries - compare keys only
        all_keys = set(t1.keys()) | set(t2.keys())
        
        for key in all_keys:
            new_path = f"{path}.{key}" if path != "root" else key
            
            if key not in t1:
                differences[new_path] = {"status": "added"}
            elif key not in t2:
                differences[new_path] = {"status": "removed"}
            else:
                # Recursively compare the structure
                compare_trees(t1[key], t2[key], new_path)
    
    compare_trees(tree1, tree2)
    return differences

def extract_nodes(tree):
    """
    Extract all node names from a tree dictionary.
    
    Args:
        tree: A tree dictionary
        
    Returns:
        A set of node names
    """
    nodes = set()
    
    def traverse(node, path=""):
        if isinstance(node, dict):
            # Add current node to the set
            if path:
                nodes.add(path)
                
            # Traverse children
            for key, value in node.items():
                new_path = f"{path}.{key}" if path else key
                traverse(value, new_path)
        else:
            # Leaf node
            if path:
                nodes.add(path)
    
    traverse(tree)
    return nodes

def generate_itol_colorstrip(differences, tree1, tree2, filename="colorstrip_annotation.txt"):
    """
    Generate an iTOL colorstrip annotation file to visualize tree differences.
    
    Args:
        differences: Dictionary of differences from identify_tree_differences
        tree1: First tree
        tree2: Second tree
        filename: Output filename for the iTOL annotation
    """
    # Extract all nodes from both trees
    nodes1 = extract_nodes(tree1)
    nodes2 = extract_nodes(tree2)
    all_nodes = nodes1.union(nodes2)
    
    # Set up colors for different types of changes
    colors = {
        "added": "#00FF00",             # Green
        "removed": "#FF0000",           # Red
        "structure_changed": "#0000FF", # Blue
        "unchanged": "#CCCCCC"          # Grey
    }
    
    # Create a colorstrip annotation
    with open(filename, "w") as f:
        # Write the header
        f.write("DATASET_COLORSTRIP\n")
        f.write("SEPARATOR TAB\n")
        f.write("DATASET_LABEL\tTree Structure Differences\n")
        f.write("COLOR\t#ff0000\n")
        f.write("COLOR_BRANCHES\t1\n")
        
        # Write the legend
        f.write("LEGEND_TITLE\tNode Status\n")
        f.write("LEGEND_SHAPES\t1\t1\t1\t1\n")
        f.write(f"LEGEND_COLORS\t{colors['added']}\t{colors['removed']}\t{colors['structure_changed']}\t{colors['unchanged']}\n")
        f.write("LEGEND_LABELS\tAdded\tRemoved\tStructure Changed\tUnchanged\n")
        
        # Write the data
        f.write("DATA\n")
        
        # Process each node
        for node in all_nodes:
            node_name = node.replace(".", "_")  # Format for iTOL
            
            if node in differences:
                status = differences[node]["status"]
                color = colors.get(status, colors["structure_changed"])
                label = f"{status.capitalize().replace('_', ' ')}: {node}"
            else:
                color = colors["unchanged"]
                label = f"Unchanged: {node}"
                
            # Use explicit tab character '\t' between fields
            f.write(f"{node_name}\t{color}\t{label}\n")

def generate_itol_branch_annotation(differences, tree1, tree2, filename="branch_annotation.txt"):
    """
    Generate an iTOL branch annotation file to visualize tree differences.
    
    Args:
        differences: Dictionary of differences from identify_tree_differences
        tree1: First tree
        tree2: Second tree
        filename: Output filename for the iTOL annotation
    """
    # Set up colors for different types of changes
    colors = {
        "added": "#00FF00",               # Green
        "removed": "#FF0000",             # Red
        "structure_changed": "#0000FF",   # Blue
    }
    
    # Create a branch annotation
    with open(filename, "w") as f:
        # Write the header
        f.write("TREE_COLORS\n")
        f.write("SEPARATOR TAB\n")
        f.write("DATA\n")
        
        # Process each difference
        for node, diff in differences.items():
            node_name = node.replace(".", "_")  # Format for iTOL
            status = diff["status"]
            
            # Get appropriate color
            if status in colors:
                color = colors[status]
            else:
                color = colors["structure_changed"]  # Default to structure_changed
            
            # Branch range (whole branch)
            f.write(f"{node_name}\tbranch\t{color}\tnormal\t1\n")

def generate_itol_label_annotation(differences, filename="label_annotation.txt"):
    """
    Generate an iTOL label annotation file to visualize tree differences.
    
    Args:
        differences: Dictionary of differences from identify_tree_differences
        filename: Output filename for the iTOL annotation
    """
    # Create a label annotation
    with open(filename, "w") as f:
        # Write the header
        f.write("LABELS\n")
        f.write("SEPARATOR TAB\n")
        f.write("DATA\n")
        
        # Process each difference
        for node, diff in differences.items():
            node_name = node.replace(".", "_")  # Format for iTOL
            status = diff["status"]
            
            # Create a descriptive label based only on structure
            if status == "structure_changed":
                details = diff.get("details", "")
                label = f"{node}: Structure changed ({details})"
            elif status == "added":
                label = f"{node}: Node added"
            elif status == "removed":
                label = f"{node}: Node removed"
            else:
                label = f"{node}: {status}"
                
            f.write(f"{node_name}\t{label}\n")

def visualize_tree_differences_for_itol(tree1, tree2, output_dir="itol_annotations"):
    """
    Generate all iTOL annotation files to visualize differences between two trees.
    
    Args:
        tree1: First tree (dictionary)
        tree2: Second tree (dictionary)
        output_dir: Directory to save the annotation files
    
    Returns:
        Dictionary of paths to generated files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Identify differences
    differences = identify_tree_differences(tree1, tree2)
    
    # Generate annotation files
    colorstrip_path = os.path.join(output_dir, "colorstrip_annotation.txt")
    branch_path = os.path.join(output_dir, "branch_annotation.txt")
    label_path = os.path.join(output_dir, "label_annotation.txt")
    
    generate_itol_colorstrip(differences, tree1, tree2, colorstrip_path)
    generate_itol_branch_annotation(differences, tree1, tree2, branch_path)
    generate_itol_label_annotation(differences, label_path)
    
    # Save the differences as JSON for reference
    diff_path = os.path.join(output_dir, "differences.json")
    with open(diff_path, "w") as f:
        json.dump(differences, f, indent=2)
    
    return {
        "colorstrip": colorstrip_path,
        "branch": branch_path,
        "label": label_path,
        "differences": diff_path
    }

def generate_newick_from_dict(tree, root_name="Tree"):
    """
    Generate a simplified Newick format string from a dictionary tree.
    This is a basic implementation and might need adjustments for complex trees.
    
    Args:
        tree: A dictionary representing a tree
        root_name: Name of the root node
    
    Returns:
        A Newick format string
    """
    def traverse(node, name):
        if isinstance(node, dict):
            # Internal node with children
            children = []
            for key, value in node.items():
                child_name = key
                child_newick = traverse(value, child_name)
                children.append(str(child_newick))
            
            if children:
                return f"({','.join(children)}){name}"
            else:
                return name
        else:
            # Leaf node
            return name
    
    newick = traverse(tree, root_name) + ";"
    return newick

def save_trees_as_newick(tree1, tree2, output_dir="itol_annotations"):
    """
    Save both trees in Newick format.
    
    Args:
        tree1: First tree (dictionary)
        tree2: Second tree (dictionary)
        output_dir: Directory to save the Newick files
    
    Returns:
        Dictionary with paths to the Newick files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate Newick strings
    newick1 = generate_newick_from_dict(tree1, "Tree1")
    newick2 = generate_newick_from_dict(tree2, "Tree2")
    
    # Save to files
    tree1_path = os.path.join(output_dir, "tree1.nwk")
    tree2_path = os.path.join(output_dir, "tree2.nwk")
    
    with open(tree1_path, "w") as f:
        f.write(newick1)
    
    with open(tree2_path, "w") as f:
        f.write(newick2)
    
    return {
        "tree1": tree1_path,
        "tree2": tree2_path
    }

# Example usage
def generate_itol_trees(undamaged_tree, damaged_tree):
    # Example trees
    
    # Generate iTOL annotation files
    output_dir = "itol_annotations"
    annotation_files = visualize_tree_differences_for_itol(undamaged_tree, damaged_tree, output_dir)
    
    # Save trees in Newick format
    newick_files = save_trees_as_newick(undamaged_tree, damaged_tree, output_dir)
    
    # Print paths to generated files
    print("Generated files:")
    for file_type, path in {**annotation_files, **newick_files}.items():
        print(f"- {file_type}: {path}")