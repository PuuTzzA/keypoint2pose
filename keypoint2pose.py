import bpy
import os
import numpy as np
import math
import mathutils
import random

# Set the path to your OBJ file
relative_obj_file_path = os.path.join("//test", "carModel.obj")
obj_file_path = bpy.path.abspath(relative_obj_file_path)
relative_points_file_path = os.path.join("//test", "test_ndc_coordinates.txt")
points_file_path = bpy.path.abspath(relative_points_file_path)

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
    [0, 0, 1, 9],
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

# define some helper functions to better visualize the data
def set_camera_params():
    camera = bpy.context.scene.camera
    focal_length = camera.data.lens
    sensor_width = camera.data.sensor_width
    n = camera.data.clip_start
    f = camera.data.clip_end
    
    render = bpy.context.scene.render
    resolution = (render.resolution_x, render.resolution_y)

    return focal_length, sensor_width, n, f, resolution

def compute_frustum(focal_length, sensor_width, near, resolution):
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera/implementing-virtual-pinhole-camera.html
    fov = 2 * math.atan(sensor_width / (2 * focal_length))

    r = near * math.tan(fov / 2)
    l = -r

    sensor_height = sensor_width * resolution[1] / resolution[0]
    fov_vertical = 2 * math.atan(sensor_height / (2 * focal_length))

    t = near * math.tan(fov_vertical / 2)
    b = -t

    return l, r, t, b, fov, fov_vertical

def create_view_frustum_visualizer(l, r, t, b, n, f, fov, fov_vertical):
    # Create a new cube
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    cube.name = "ViewFrustum"

    # Enter edit mode to modify the vertices
    #bpy.ops.object.mode_set(mode='EDIT')
    #bpy.ops.mesh.select_all(action='SELECT')

    # Access the mesh data
    mesh = cube.data

    # Set the vertex coordinates
    r_far = f * math.tan(fov / 2)
    l_far = -r_far
    t_far = f * math.tan(fov_vertical / 2)
    b_far = -t_far
    vertex_coords = [
        (r, t, n),
        (r, b, n),
        (l, t, n),
        (l, b, n),
        (r_far, t_far, f),
        (r_far, b_far, f),
        (l_far, t_far, f),
        (l_far, b_far, f),
    ]

    for i, coord in enumerate(vertex_coords):
       mesh.vertices[i].co = coord

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add and apply the Wireframe modifier
    cube.modifiers.new(name="Wireframe", type='WIREFRAME')
    #bpy.ops.object.modifier_apply(modifier=wireframe_modifier.name)

    bpy.ops.mesh.primitive_cube_add(size=2)
    cube = bpy.context.object
    cube.name = "NDC"
    cube.modifiers.new(name="Wireframe", type='WIREFRAME')

def create_line(start, end, thickness=0.01):
    length = (mathutils.Vector(end) - mathutils.Vector(start)).length
    
    # Create a cylinder and set its dimensions
    bpy.ops.mesh.primitive_cylinder_add(radius=thickness, depth=length, location=(0, 0, 0))
    line = bpy.context.object
    
    # Set the position of the cylinder to be between the start and end points
    line.location = [(s + e) / 2 for s, e in zip(start, end)]
    
    # Calculate the direction vector and rotation of the line
    direction = mathutils.Vector(end) - mathutils.Vector(start)
    rot_quat = direction.to_track_quat('Z', 'Y')
    line.rotation_euler = rot_quat.to_euler()
    
def dis(a, b):
    sum = 0
    try:
        for i in range(len(a)):
            sum += (a[i] - b[i]) * (a[i] - b[i])
    except:
        print("dimension mismach ", len(a), " is not ", len(b))
    return math.sqrt(sum)

# setup camera and projection matrix
n = 1
f = 15

focal_length = 30
sensor_width = 36
resolution = (1920, 1080)

focal_length, sensor_width, n, f, resolution = set_camera_params()
l, r, t, b, fov, fov_vertical = compute_frustum(focal_length, sensor_width, n, resolution)

create_view_frustum_visualizer(l, r, t, b, n, f, fov, fov_vertical)

m_p = np.array([
    [n, 0, 0, 0],
    [0, n, 0, 0],
    [0, 0, n+f, -n*f],
    [0, 0, 1, 0]
])

m_ndc = np.array([
    [2/(r - l), 0, 0, -(r + l)/(r - l)],
    [0, 2/(t - b), 0, -(t + b)/(t - b)],
    [0, 0, 2/(f - n), -(f + n)/(f - n)],
    [0, 0, 0, 1]
])

projection_matrix = m_ndc @ m_p

# Read in the OBJ
POINTS_FROM_FILE = False #if True then use points from file, if False use Test model matrix
VISUALIZE = True #if True, visualize everything
vertices_3d = []
points_2d = []

points_2d_ground_truth = []
vertices_3d_world_coords = []

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
                bpy.ops.mesh.primitive_cube_add(size=0.1, location=(location[0], location[1], location[2]))
                vertices_3d_world_coords.append([location[0], location[1], location[2]])
                
                location = projection_matrix @ location
                location /= location[3]

                bpy.ops.mesh.primitive_cube_add(size=0.1, location=(location[0], location[1], location[2]))

                points_2d_ground_truth.append([location[0], location[1]])

                bounds = 0.05
                points_2d.append([location[0] + random.uniform(-bounds, bounds), location[1] + random.uniform(-bounds, bounds)])

points_2d[3][0] += 0.09
points_2d[3][1] -= 0.2

if POINTS_FROM_FILE:
    with open(points_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            x_, y_ = float(parts[0]), float(parts[1])
            points_2d.append([x_, y_])

# Direct Linear Transform Estimator model
def direct_linear_transform(sample):
    A = []
    b = []

    p_matrix = sample["projection_matrix"]

    c1 = p_matrix[0][0]
    c2 = p_matrix[1][1]
    c3 = p_matrix[2][2]
    c4 = p_matrix[2][3]

    for s in sample["points"]:
        x, y, z = s["3d"]
        x_, y_ = s["2d"]

        A.append([-c1 * x, -c1 * y, -c1 * z, -c1, 0, 0, 0, 0, x * x_, y * x_, z * x_, x_])
        A.append([0, 0, 0, 0, -c2 * x, -c2 * y, -c2 * z, -c2, x * y_, y * y_, z * y_, y_])
        b.append(0)
        b.append(0)

    A = np.array(A)
    b = np.array(b)

    U, S, Vt = np.linalg.svd(A)

    model_matrix_predicted = Vt[-1].reshape(3, 4)
    model_matrix_predicted /= model_matrix_predicted[-1][-1]
    model_matrix_predicted = np.vstack([model_matrix_predicted, [0, 0, 0, 1]])

    # get the translation and rotation part out of the prediciton
    R = [[model_matrix_predicted[i][j] for j in range(0, 3)] for i in range(0, 3)]
    T = [model_matrix_predicted[i][3] for i in range(0, 3)]
    R = np.array(R)
    T = np.array(T)

    # force the scale to be 1
    U, _, Vt = np.linalg.svd(R)
    R_orthogonal = U @ Vt

    # rescale the translation by the same amount
    s = R_orthogonal[0][0] / R[0][0]
    T *= s

    model_matrix_predicted = np.eye(4)

    # put the rotation and translation back together
    for i in range(0, 4):
        for j in range(0, 4):
            if i < 3 and j < 3:
                model_matrix_predicted[i][j] = R_orthogonal[i][j]
            elif i < 3:
                model_matrix_predicted[i][j] = T[i]
    
    return model_matrix_predicted

def dlt_loss(model, point, projection_matrix):
    x, y, z = point["3d"]
    x_, y_ = point["2d"]

    projected = projection_matrix @ model @ np.array([x, y, z, 1])
    projected /= projected[3]

    return dis([projected[0], projected[1]], [x_, y_])

class RANSAC:
    def __init__(self, n=8, k=500, t=0.1, d=7, model=None, loss=None):
        self.n = n              # `n`: Minimum number of data points to estimate parameters
        self.k = k              # `k`: Maximum iterations allowed
        self.t = t              # `t`: Threshold value to determine if points are fit well
        self.d = d              # `d`: Number of close data points required to assert model fits well
        self.model = model      # `model`: function that takes a sample and predicts a model
        self.loss = loss        # `loss`: function that calculates the error of one point
        self.best_model = None
        self.best_error = np.inf


    def reset(self):
        self.best_model = None
        self.best_error = np.inf

    def fit(self, data):
        self.reset()
        best_inliers = []
        projection_matrix = data["projection_matrix"]

        for _ in range(self.k):
            sample = random.sample(data["points"], self.n)

            estimation = self.model({"points" : sample, "projection_matrix" : projection_matrix})

            inliers = []
            for point in data["points"]:
                error = self.loss(estimation, point, projection_matrix)
                if error < self.t:
                    inliers.append(point)
            
            print(len(inliers))
            if len(inliers) >= self.d:
                better_estimation = self.model({"points" : inliers, "projection_matrix" : projection_matrix})
                total_error = sum(self.loss(better_estimation, p, projection_matrix) for p in data["points"])
                
                if total_error < self.best_error:
                    self.best_model = better_estimation
                    self.best_error = total_error
                    best_inliers = inliers

        if self.best_error == np.inf:
            print("RANSAC failed")
            return self.model(data), []
        
        return self.best_model, best_inliers

#def dlt_error(model, )

data = {"points" : [{"3d": v, "2d": p} for v, p in zip(vertices_3d, points_2d)], "projection_matrix" : projection_matrix}


regressor = RANSAC(model=direct_linear_transform, loss=dlt_loss)
mm, inliers = regressor.fit(data)

#mm = direct_linear_transform(data)

print("model_matrix:")
print(model_matrix)
print("predicted model matrix")
print(mm)
print("difference:")
print(mm - model_matrix)
print("difference in the position of the points:")
print("point,\t screenspace_dif,\t worldspace_dif")

for i in range(0, len(vertices_3d)):
    x, y, z = vertices_3d[i]
    x_, y_ = points_2d[i]

    location = np.array([x, y, z, 1])
    location = mm @ location
    bpy.ops.mesh.primitive_cube_add(size=0.6, location=(location[0], location[1], location[2]))

    # Visualization
    start_point = np.array([x_, y_, -1, 1])
    end_point = np.array([x_, y_, 1, 1])
    
    create_line(start_point[:3], end_point[:3])
    
    inverse_projection_matrix = np.linalg.inv(projection_matrix)
    
    start_point *= n
    start_point = inverse_projection_matrix @ start_point
    end_point *= f
    end_point = inverse_projection_matrix @ end_point
    
    create_line(start_point[:3], end_point[:3])

    print(i, ":\t", dis(points_2d[i], points_2d_ground_truth[i]) , "\t", dis([location[0], location[1], location[2]], vertices_3d_world_coords[i]))

print("----------------------------------")   