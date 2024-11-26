import bpy
import mathutils
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from algorithms import *

class PoseApproximation:
    def __init__(self, vertices_3d, projection_matrix, obj_file_path, ransac_n=8, ransac_d=6):
        self.vertices_3d = vertices_3d
        self.projection_matrix = projection_matrix
        self.model = None
        self.inliers = []
        self.points_2d = []
        self.regressor = RANSAC(model=direct_linear_transform, loss=dlt_loss, n=ransac_n, d=ransac_d)

        # import the obj and assign the wireframe geonodes graph
        bpy.ops.wm.obj_import(filepath=obj_file_path)
        self.obj = bpy.context.selected_objects[0]
        geo_modifier = self.obj.modifiers.new(name="WireframeGeoNodes", type='NODES')
        node_group_name = "wireframe"
        if node_group_name in bpy.data.node_groups:
            geo_modifier.node_group = bpy.data.node_groups[node_group_name]
        else:
            print(f"Geometry Nodes group '{node_group_name}' not found!")

    def fit(self, points_2d):   
        self.points_2d = points_2d
        data = {"points" : [{"3d": v, "2d": p} for v, p in zip(self.vertices_3d, points_2d)], "projection_matrix" : self.projection_matrix}
        self.model, self.inliers = self.regressor.fit(data)

    def visualize(self, line_start=2, line_end=30):
        # move the obj to the right spot
        blender_matrix = mathutils.Matrix([list(row) for row in self.model])
        self.obj.matrix_world = blender_matrix

        # indicate inliers
        for p in self.inliers:
            x, y, z = p["3d"]
            location = self.model @ np.array([x, y, z, 1])

            bpy.ops.mesh.primitive_cube_add(size=0.6, location=(location[0], location[1], location[2]))
            inlier = bpy.context.selected_objects[0]

            # Add a Geometry Nodes modifier and add the "wireframe" node group
            geo_modifier = inlier.modifiers.new(name="WireframeGeoNodes", type='NODES')
            node_group_name = "inlier"
            if node_group_name in bpy.data.node_groups:
                geo_modifier.node_group = bpy.data.node_groups[node_group_name]
            else:
                print(f"Geometry Nodes group '{node_group_name}' not found!")

        # add 2d point "indicators"
        print("i:\t", "distance(NDC)")
        for i in range(len(self.points_2d)):    
            x, y, z = self.vertices_3d[i]
            x_, y_ = self.points_2d[i]   

            start_point = np.array([x_, y_, -1, 1])
            end_point = np.array([x_, y_, 1, 1])
            PoseApproximation.create_line(start_point[:3], end_point[:3])
            
            inverse_projection_matrix = np.linalg.inv(self.projection_matrix)
            start_point *= line_start
            start_point = inverse_projection_matrix @ start_point
            end_point *= line_end
            end_point = inverse_projection_matrix @ end_point 
            PoseApproximation.create_line(start_point[:3], end_point[:3])
        
            projected = self.projection_matrix @ self.model @ np.array([x, y, z, 1])
            projected /= projected[3]
            print(str(i) + ": \t", dis(self.points_2d[i], projected[:2]))
        
    def create_line(start, end, thickness=0.0075):
        length = (mathutils.Vector(end) - mathutils.Vector(start)).length

        # Create a cylinder and set its dimensions
        bpy.ops.mesh.primitive_cylinder_add(radius=thickness, depth=length, location=(0, 0, 0))
        line = bpy.context.object
        mat = bpy.data.materials.get("2d_points")
        line.data.materials.append(mat)

        # Set the position of the cylinder to be between the start and end points
        line.location = [(s + e) / 2 for s, e in zip(start, end)]

        # Calculate the direction vector and rotation of the line
        direction = mathutils.Vector(end) - mathutils.Vector(start)
        rot_quat = direction.to_track_quat('Z', 'Y')
        line.rotation_euler = rot_quat.to_euler()