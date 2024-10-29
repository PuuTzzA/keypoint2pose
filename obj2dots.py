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

model_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [0, 0, 1, 9],
    [0, 0, 0, 1]
])


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
            
            location2 = location.copy()

            location = m_p_naive @ location
            location /= location[3]

            #bpy.ops.mesh.primitive_cube_add(size=0.01, location=(location[0], location[1], location[2]))

            location2 = m_o @ m_p @ location2

            x = location2[3]
            location2 /= location2[3]

            bpy.ops.mesh.primitive_cube_add(size=0.1, location=(location2[0], location2[1], location2[2]))


with open(points_file_path, 'r') as file:
    for line in file:

        parts = line.strip().split(',')
        print(parts[0])      

        x, y = float(parts[0]), float(parts[1])
        start_point = np.array([x, y, -1, 1])
        end_point = np.array([x, y, 1, 1])

        create_line(start_point[:3], end_point[:3])

        inverse_projection_matrix = np.linalg.inv(m_o @ m_p)

        start_point *= n
        start_point = inverse_projection_matrix @ start_point
        end_point *= f
        end_point = inverse_projection_matrix @ end_point

        create_line(start_point[:3], end_point[:3])