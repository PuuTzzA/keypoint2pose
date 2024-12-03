import bpy
import os
import numpy as np
import math
import mathutils
import json
import random
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import importlib
import source.scene as scene
import source.approximation as approximation
import source.algorithms as algorithms
importlib.reload(scene)
importlib.reload(approximation)
importlib.reload(algorithms)
from source.scene import Scene
from source.algorithms import * 
from source.approximation import * 

# Set the path to your OBJ file
current_dir = os.path.dirname(bpy.data.filepath)  # Blender's current working directory

obj_file_path = os.path.join(current_dir, "3d_model/nissan_altima_wireframe.obj")
obj_keypoints_file_path = os.path.join(current_dir, "3d_model/nissan_altima_keypoints.obj")
json_file_path = os.path.join(current_dir, "test/real_data_test.json")

if not os.path.exists(obj_file_path) or not os.path.exists(obj_keypoints_file_path):
    print("One or more files not found:", obj_file_path)
else:
    print("Files found, proceeding...")

# Clear all existing mesh objects in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='CURVE')
bpy.ops.object.delete()

# setup camera and projection matrix
first_image_file_path = os.path.join(current_dir, "test/real_data_test.jpg")

image = bpy.data.images.load(first_image_file_path)
scene = Scene(resolution=image.size, n=2, f=200)
scene.setup_blender_scene()

projection_matrix = scene.get_projection_matrix()

vertices_3d = []

with open(obj_keypoints_file_path, 'r') as file:
    for line in file:
        if line.startswith('v '):
            parts = line.strip().split()
            x, y, z = [float(parts[1]), float(parts[2]), float(parts[3])]
            vertices_3d.append([x, y, z])

for e in vertices_3d:
    print(e)

json_data = None
with open(json_file_path, "r") as file:
    json_data = json.load(file)


print(len(json_data[0]["keypoints"]))

for j in range(len(json_data)):

    if j != 1:
        continue

    car = PoseApproximation(vertices_3d=vertices_3d, projection_matrix=projection_matrix, obj_file_path=obj_file_path)

    print("amount of keypoints: ", len(json_data[j]["keypoints"]) / 3)
    points_2d = []

    for i in range(len(json_data[j]["keypoints"]) // 3):    
        x_, y_, confidence = json_data[j]["keypoints"][i * 3: i * 3 + 3]
        x_, y_ = scene.pixel_to_NDC(x_, y_)

        points_2d.append({"point": [x_, y_], "confidence": confidence})


        start_point = np.array([x_, y_, -1, 1])
        end_point = np.array([x_, y_, 1, 1])

        PoseApproximation.create_line(start_point[:3], end_point[:3])

        #inverse_projection_matrix = np.linalg.inv(projection_matrix)
        #start_point *= 2
        #start_point = inverse_projection_matrix @ start_point
        #end_point *= 30
        #end_point = inverse_projection_matrix @ end_point 
        #PoseApproximation.create_line(start_point[:3], end_point[:3])

    car.fit(points_2d=points_2d)
    car.visualize(scene.n * 1.1, scene.f)

scene.set_background(os.path.join(current_dir, "test/real_data_test.jpg"))
scene.render_frame(os.path.join(current_dir, "output/overlayed_image.png"))

print("----------------------------------")   