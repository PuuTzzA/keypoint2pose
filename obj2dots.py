import bpy
import os
import numpy as np
import math
import mathutils


# Set the path to your OBJ file
relative_obj_file_path = os.path.join("//test", "test_cube.obj")
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

translation = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [0, 0, 1, 9],
    [0, 0, 0, 1]
])

r_x = math.pi/8
r_y = math.pi/4
r_z = 1.223422

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


def set_camera_params():
    camera = bpy.context.scene.camera
    focal_length = camera.data.lens
    sensor_width = camera.data.sensor_width
    n = camera.data.clip_start
    f = camera.data.clip_end
    
    render = bpy.context.scene.render
    resolution = (render.resolution_x, render.resolution_y)

    return focal_length, sensor_width, n, f, resolution


# https://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera/implementing-virtual-pinhole-camera.html
def compute_frustum(focal_length, sensor_width, near, resolution):
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
    

n = 1
f = 15

focal_length = 30
sensor_width = 36
resolution = (1920, 1080)


focal_length, sensor_width, n, f, resolution = set_camera_params()
l, r, t, b, fov, fov_vertical = compute_frustum(focal_length, sensor_width, n, resolution)


create_view_frustum_visualizer(l, r, t, b, n, f, fov, fov_vertical)


m_p_naive = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0]
])

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

vertices_3d = []
POINTS_FROM_FILE = False
points_2d = []

# Open the OBJ file and read line by line
with open(obj_file_path, 'r') as file:
    for line in file:
        # OBJ vertices start with "v "
        if line.startswith('v '):
            # Extract x, y, z coordinates
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            vertices_3d.append([x, y, z])

            location = np.array([x, y, z, 1])
            location = model_matrix @ location
            
            # Place a cube at each vertex position
            bpy.ops.mesh.primitive_cube_add(size=0.5, location=(location[0], location[1], location[2]))
            
            location2 = location.copy()

            location = m_p_naive @ location
            location /= location[3]

            #bpy.ops.mesh.primitive_cube_add(size=0.01, location=(location[0], location[1], location[2]))

            location2 = projection_matrix @ location2

            location2 /= location2[3]

            bpy.ops.mesh.primitive_cube_add(size=0.1, location=(location2[0], location2[1], location2[2]))

            if not POINTS_FROM_FILE:
                points_2d.append([location2[0], location2[1]])





A = np.zeros((2 * len(vertices_3d), 12))
b = np.zeros(2 * len(vertices_3d))

if POINTS_FROM_FILE:
    with open(points_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')

            x_, y_ = float(parts[0]), float(parts[1])

            points_2d.append([x_, y_])



print("----------------------------------")

for i in range(0, len(vertices_3d)):
    c1 = projection_matrix[0][0]
    c2 = projection_matrix[1][1]
    c3 = projection_matrix[2][2]
    c4 = projection_matrix[2][3]

    x_ = points_2d[i][0]
    y_ = points_2d[i][1]
    x = vertices_3d[i][0]
    y = vertices_3d[i][1]
    z = vertices_3d[i][2]

    A[2 * i] = [-c1 * x, -c1 * y, -c1 * z, -c1, 0, 0, 0, 0, x * x_, y * x_, z * x_, x_]
    A[2 * i + 1] = [0, 0, 0, 0, -c2 * x, -c2 * y, -c2 * z, -c2, x * y_, y * y_, z * y_, y_]

U, S, Vt = np.linalg.svd(A)

model_matrix_predicted = Vt[-1].reshape(3, 4)
model_matrix_predicted /= model_matrix_predicted[-1][-1]
model_matrix_predicted = np.vstack([model_matrix_predicted, [0, 0, 0, 1]])

R = [[model_matrix_predicted[i][j] for j in range(0, 3)] for i in range(0, 3)]
T = [model_matrix_predicted[_][3] for _ in range(0, 3)]
R = np.array(R)
T = np.array(T)

U, _, Vt = np.linalg.svd(R)
R_orthogonal = U @ Vt

s = R_orthogonal[0][0] / R[0][0]
T *= s

model_matrix_predicted_refined = np.eye(4)

for i in range(0, 4):
    for j in range(0, 4):
        if i < 3 and j < 3:
            model_matrix_predicted_refined[i][j] = R_orthogonal[i][j]
        elif i < 3:
            model_matrix_predicted_refined[i][j] = T[i]

#refined_model_matrix[:3][:3] = R_orthogonal[:3][:3]
#refined_model_matrix[:3][3] = T


print("model_matrix:")
print(model_matrix)
#print("model_matrix_predicted")
#print(model_matrix_predicted)
print("model_matrix_predicted_refined")
print(model_matrix_predicted_refined)
print("difference:")
print(model_matrix_predicted_refined - model_matrix)


for i in range(0, len(vertices_3d)):
    x, y, z = vertices_3d[i]
    x_, y_ = points_2d[i]

    location = np.array([x, y, z, 1])
    location = model_matrix_predicted_refined @ location
    bpy.ops.mesh.primitive_cube_add(size=1, location=(location[0], location[1], location[2]))

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

print("----------------------------------")