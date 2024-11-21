import bpy
import os
import numpy as np
import random

# Set the paths to your files
relative_obj_file_path = os.path.join("//test", "carModel.obj")
obj_file_path = bpy.path.abspath(relative_obj_file_path)
relative_points_file_path = os.path.join("//test", "test_ndc_coordinates.txt")
points_file_path = bpy.path.abspath(relative_points_file_path)

if not os.path.exists(obj_file_path) or not os.path.exists(points_file_path):
    print("One or more files not found:", obj_file_path, points_file_path)
else:
    print("Files found, proceeding...")

# Clear all existing mesh objects in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Define RANSAC function
def ransac(data, estimate_model, compute_error, n=4, k=100, t=0.01, d=10):
    best_model = None
    best_inliers = []
    best_error = float('inf')

    for _ in range(k):
        # Randomly sample `n` points
        sample = random.sample(data, n)
        
        # Fit a model to the sample
        model = estimate_model(sample)
        
        # Determine inliers
        inliers = []
        for point in data:
            error = compute_error(model, point)
            if error < t:
                inliers.append(point)
        
        # Update the best model if conditions are met
        if len(inliers) > d:
            total_error = sum(compute_error(model, p) for p in inliers)
            if total_error < best_error:
                best_model = model
                best_inliers = inliers
                best_error = total_error

    return best_model, best_inliers

# Estimation function for the model
def estimate_model(sample):
    A = []
    b = []
    for s in sample:
        x, y, z = s['3d']
        x_, y_ = s['2d']
        A.append([x, y, z, 1, 0, 0, 0, 0, -x * x_, -y * x_, -z * x_, -x_])
        A.append([0, 0, 0, 0, x, y, z, 1, -x * y_, -y * y_, -z * y_, -y_])
        b.extend([0, 0])
    
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 4)

# Error function
def compute_error(model, point):
    x, y, z = point['3d']
    x_, y_ = point['2d']
    projected = model @ np.array([x, y, z, 1])
    projected /= projected[3]
    return np.linalg.norm([projected[0] - x_, projected[1] - y_])

# Load 3D and 2D points
vertices_3d = []
points_2d = []

with open(obj_file_path, 'r') as obj_file:
    for line in obj_file:
        if line.startswith('v '):
            parts = line.split()
            vertices_3d.append([float(parts[1]), float(parts[2]), float(parts[3])])

with open(points_file_path, 'r') as points_file:
    for line in points_file:
        parts = line.split(',')
        points_2d.append([float(parts[0]), float(parts[1])])

data = [{'3d': v, '2d': p} for v, p in zip(vertices_3d, points_2d)]

# Run RANSAC
ransac_model, inliers = ransac(data, estimate_model, compute_error)

# Output results
print("Estimated Transformation Matrix:")
print(ransac_model)
print(f"Number of Inliers: {len(inliers)}")
