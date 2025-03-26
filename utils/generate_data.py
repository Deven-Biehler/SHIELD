import laspy
import numpy as np



def random_sample_las(input_las_path, output_las_path, sample_fraction=0.1):
    # Read LAS file
    las = laspy.read(input_las_path)
    
    # Random sample points
    num_points = len(las.points)
    sample_size = int(num_points * sample_fraction)
    sample_indices = np.random.choice(num_points, sample_size, replace=False)
    las.points = las.points[sample_indices]
    
    las.write(output_las_path)

def bounding_box_sample_las(input_las_path, output_las_path, bounding_box):
    
    # Extract bounding box coordinates
    min_x, max_x, min_y, max_y, min_z, max_z = bounding_box
    


    chunk_size = 1000000  # Adjust chunk size as needed
    with laspy.open(input_las_path) as f:
        with laspy.open(output_las_path, mode="w", header=f.header) as writer:
            for points in f.chunk_iterator(chunk_size):
                x, y, z = points.x, points.y, points.z
                in_box_indices = (
                    (x >= min_x) & (x <= max_x) &
                    (y >= min_y) & (y <= max_y) &
                    (z >= min_z) & (z <= max_z)
                )
                writer.write_points(points[in_box_indices])
    
    


def generate_damage(input_las_path, output_las_path, location, aggressiveness):
    '''
    Takes a las file and simulates real world damage by removing sections

    params:
    input_las_path: las file path
    output_las_path: output las file path
    location: location of the damage (x, y, z)
    aggressiveness: size of damage (radius)
    '''
    
    
    chunk_size = 1000000  # Adjust chunk size as needed
    with laspy.open(input_las_path) as f:
        with laspy.open(output_las_path, mode="w", header=f.header) as writer:
            for points in f.chunk_iterator(chunk_size):
                x, y, z = points.x, points.y, points.z
                distance = np.sqrt((x - location[0])**2 + (y - location[1])**2 + (z - location[2])**2)
                non_damaged_indices = distance > aggressiveness
                writer.write_points(points[non_damaged_indices])



if __name__ == "__main__":
    lidar_path = "data/lidar_data/AVST0069_A10.laz"
    
    
    bounding_box_sampled_las = "data/damage_detection/house.laz"
    point1 = (2436185.927, 230199.978, 2415.064)
    point2 = (2436427.024, 230443.331, 2414.181)
    x_bounds = (min(point1[0], point2[0]), max(point1[0], point2[0]))
    y_bounds = (min(point1[1], point2[1]), max(point1[1], point2[1]))
    z_bounds = (min(point1[2], point2[2])-1000, max(point1[2], point2[2])+1000)
    bounding_box = (*x_bounds, *y_bounds, *z_bounds)
    bounding_box_sample_las(lidar_path, bounding_box_sampled_las, bounding_box)


    output_lidar_path = "data/damage_detection/house_damaged.laz"
    damage_xyz = (2436371.258, 230351.648, 2432.160)
    damage_radius = 8.5
    generate_damage(bounding_box_sampled_las, output_lidar_path, damage_xyz, damage_radius)


    damaged_las = output_lidar_path[:-4] + "_rsample0.5.laz"
    undamaged_las = bounding_box_sampled_las[:-4] + "_rsample0.5.laz"
    random_sample_las(bounding_box_sampled_las, undamaged_las, 0.5)
    random_sample_las(output_lidar_path, damaged_las, 0.5)
    


