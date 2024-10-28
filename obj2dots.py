import bpy
import os
import numpy as np
import math

# Set the path to your OBJ file
obj_file_path = r"C:\__Coding__\SNU_5_Semester\Robot_Vision\Project\3d_reconstruction\keypoint2pose\test\test_cube.obj"

if not os.path.exists(obj_file_path):
    print("File not found:", obj_file_path)
else:
    print("File found, proceeding...")


# Clear all existing mesh objects in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

model_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 9],
    [0, 0, 1, -1],
    [0, 0, 0, 1]
])


# https://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera/implementing-virtual-pinhole-camera.html
def compute_frustum(focal_length, sensor_width, near, resolution):
    fov = 2 * math.atan(sensor_width / (2 * focal_length))

    r = near * math.tan(fov / 2)
    l = -r

    sensor_height = sensor_width * resolution[1] / resolution[0]
    fov_vertical = 2 * math.atan(sensor_height / (2 * focal_length))

    t = near * math.tan(fov_vertical / 2)
    b = -t

    return l, r, t, b

n = 1
f = 10

focal_length = 30
sensor_width = 36
resolution = (1920, 1080)

l, r, t, b = compute_frustum(focal_length, sensor_width, n, resolution)

bpy.ops.mesh.primitive_cube_add(size=0.2, location=(r, n, t))
bpy.ops.mesh.primitive_cube_add(size=0.2, location=(l, n, b))


m_p = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
])

m_o = np.array([
    [2/(r - l), 0, 0, -(r + l)/(r - l)],
    [0, 2/(t - b), 0, -(t + b)/(t - b)],
    [0, 0, 2/(f - n), -(f + n)/(f - n)],
    [0, 0, 0, 1]
])

projection_matrix = m_o @ m_p

# Open the OBJ file and read line by line
with open(obj_file_path, 'r') as file:
    for line in file:
        # OBJ vertices start with "v "
        if line.startswith('v '):
            # Extract x, y, z coordinates
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])

            location = np.array([x, y, z, 1])
            location = model_matrix @ location
            
            # Place a cube at each vertex position
            bpy.ops.mesh.primitive_cube_add(size=0.5, location=(location[0], location[1], location[2]))
            
            location = m_p @ location
            location /= location[3]

            bpy.ops.mesh.primitive_cube_add(size=0.01, location=(location[0], location[1], location[2]))


