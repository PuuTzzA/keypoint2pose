import bpy
import math
import mathutils
import os
import numpy as np

class Scene:
    def __init__(self, resolution = (1920, 1080), n = 2, f = 30, focal_lengh = 30, sensor_width = 36):
        self.resolution = resolution
        self.n = n
        self.f = f
        self.focal_length = focal_lengh
        self.sensor_width = sensor_width
        self.background = None
        self.l = -1
        self.r = -1
        self.t = -1
        self.b = -1

    def setup_blender_scene(self):  
        camera = bpy.context.scene.camera
        camera.data.lens = self.focal_length
        camera.data.sensor_width = self.sensor_width
        camera.data.clip_start = self.n
        camera.data.clip_end = self.f

        render = bpy.context.scene.render
        render.resolution_x = self.resolution[0]
        render.resolution_y = self.resolution[1]
        render.image_settings.file_format = 'PNG'

        self.l, self.r, self.t, self.b, fov, fov_vertical = Scene.compute_frustum(self.focal_length, self.sensor_width, self.n, self.resolution)

        self.create_view_frustum_and_background(self.l, self.r, self.t, self.b, self.n, self.f, fov, fov_vertical)

    def get_projection_matrix(self):
        m_p = np.array([
            [self.n, 0, 0, 0],
            [0, self.n, 0, 0],
            [0, 0, self.n+self.f, -self.n*self.f],
            [0, 0, 1, 0]
        ])

        m_ndc = np.array([
            [2/(self.r - self.l), 0, 0, -(self.r + self.l)/(self.r - self.l)],
            [0, 2/(self.t - self.b), 0, -(self.t + self.b)/(self.t - self.b)],
            [0, 0, 2/(self.f - self.n), -(self.f + self.n)/(self.f - self.n)],
            [0, 0, 0, 1]
        ])

        projection_matrix = m_ndc @ m_p
        return projection_matrix

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

    def create_view_frustum_and_background(self, l, r, t, b, n, f, fov, fov_vertical):
        # View Frustum
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.object
        cube.hide_render = True
        cube.name = "ViewFrustum"

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

        # Add and apply the Wireframe modifier
        cube.modifiers.new(name="Wireframe", type='WIREFRAME')

        # NDC Space
        bpy.ops.mesh.primitive_cube_add(size=2)
        cube = bpy.context.object
        cube.name = "NDC"
        cube.modifiers.new(name="Wireframe", type='WIREFRAME')

        # Background Image
        bpy.ops.mesh.primitive_plane_add(size=1)
        plane = bpy.context.object
        plane.name = "background"
        model_matrix = mathutils.Matrix([[r_far - l_far, 0, 0, 0], [0, t_far - b_far, 0, 0], [0, 0, 1, f - 0.001], [0, 0, 0, 1]])
        plane.matrix_world = model_matrix
        mat = bpy.data.materials.get("background")

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        for node in nodes:
            nodes.remove(node)

        texture_node = nodes.new(type="ShaderNodeTexImage")
        self.background = texture_node
        texture_node.location = (-300, 300)  # Adjust location for better organization

        output_node = nodes.new(type="ShaderNodeOutputMaterial")
        output_node.location = (300, 300)

        links.new(texture_node.outputs["Color"], output_node.inputs["Surface"])

        plane.data.materials.append(mat)

    def set_background(self, path):
        self.background.image = bpy.data.images.load(path)

    def render_frame(self, output_path):
        bpy.ops.render.render(write_still=False)
        bpy.data.images['Render Result'].save_render(filepath=output_path)
        print(f"Frame rendered and saved to {output_path}")