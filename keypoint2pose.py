import bpy
import os
import numpy as np
import math
import mathutils
import random
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import importlib
import scene
import approximation
import algorithms
importlib.reload(scene)
importlib.reload(approximation)
importlib.reload(algorithms)
from scene import Scene
from algorithms import *
from approximation import * 

# Set the path to your OBJ file
current_dir = os.path.dirname(bpy.data.filepath)  # Blender's current working directory

obj_file_path = os.path.join(current_dir, "test/carModel.obj")
points_file_path = os.path.join(current_dir, "test/test_ndc_coordinates.txt")

if not os.path.exists(obj_file_path) or not os.path.exists(points_file_path):
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

# Define Model Matrix (Translation and Rotation Matrices) for Testing
translation = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [0, 0, 1, 15],
    [0, 0, 0, 1]
])

r_x = 0
r_y = 0
r_z = 0

rotation_x = np.array([
    [1, 0, 0, 0],
    [0, math.cos(r_x), -math.sin(r_x), 0],
    [0, math.sin(r_x), math.cos(r_x), 0],
    [0, 0, 0, 1]
])

rotation_y = np.array([
    [math.cos(r_y), 0, math.sin(r_y), 0],
    [0, 1, 0, 0],
    [-math.sin(r_y), 0, math.cos(r_y), 0],
    [0, 0, 0, 1]
])

rotation_z = np.array([
    [math.cos(r_z), -math.sin(r_z), 0, 0],
    [math.sin(r_z), math.cos(r_z), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

model_matrix = translation @ rotation_z @ rotation_y @ rotation_x

# setup camera and projection matrix
first_image_file_path = os.path.join(current_dir, "test/10_27491.jpg")

resolution = (0, 0)
image = bpy.data.images.load(first_image_file_path)
scene = Scene(image.size)
scene.setup_blender_scene()

projection_matrix = scene.get_projection_matrix()

# Read in the OBJ
POINTS_FROM_FILE = False #if True then use points from file, if False use Test model matrix
VISUALIZE = True #if True, visualize everything
vertices_3d = []
points_2d = []

with open(obj_file_path, 'r') as file:
    for line in file:
        if line.startswith('v '):
            parts = line.strip().split()
            x, y, z = [float(parts[1]), float(parts[2]), float(parts[3])]
            vertices_3d.append([x, y, z])

            if not POINTS_FROM_FILE:
                location = np.array([x, y, z, 1])
                location = model_matrix @ location
                
                # Place a cube at each vertex position
                #bpy.ops.mesh.primitive_cube_add(size=0.1, location=(location[0], location[1], location[2]))
                
                location = projection_matrix @ location
                location /= location[3]

                bpy.ops.mesh.primitive_cube_add(size=0.1, location=(location[0], location[1], location[2]))

                bounds = 0.02
                points_2d.append([location[0] + random.uniform(-bounds, bounds), location[1] + random.uniform(-bounds, bounds)])

points_2d[3][0] += 0.09
points_2d[3][1] -= 0.2

if POINTS_FROM_FILE:
    with open(points_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            x_, y_ = float(parts[0]), float(parts[1])
            points_2d.append([x_, y_])

car1 = PoseApproximation(vertices_3d=vertices_3d, projection_matrix=projection_matrix, obj_file_path=obj_file_path)
car1.fit(points_2d)
car1.visualize(scene.n * 1.1, scene.f)

scene.set_background(os.path.join(current_dir, "test/test_cube.png"))
scene.render_frame(os.path.join(current_dir, "overlayed_image.png"))

print("----------------------------------")   