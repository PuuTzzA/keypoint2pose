import os
import numpy as np
import json
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from source.scene import Scene
from source.algorithms import * 
from source.approximation import * 


class Scene_manager():
    def __init__(self, obj_file_paths, keypoints_file_paths, input_path, output_path, max_distance=0.1, focal_length=30, frame_rate=30, start_index=1):
        self.obj_file_paths = obj_file_paths
        self.obj_keypoints_file_paths = keypoints_file_paths
        self.cars = []
        self.max_distance_traveled = max_distance # in NDC 
        self.input_path = input_path
        self.output_path = output_path
        self.frame_number = start_index
        self.frame_rate = frame_rate

        image = bpy.data.images.load(input_path + "0001.jpg")
        self.scene = Scene(resolution=image.size, n=2, f=200, focal_length=focal_length)
        self.scene.setup_blender_scene()

        self.projection_matrix = self.scene.get_projection_matrix()

        self.vertices_3d_list = [[] for _ in keypoints_file_paths]

        for i in range(len(self.vertices_3d_list)):
            with open(self.obj_keypoints_file_paths[i], 'r') as file:
                for line in file:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        x, y, z = [float(parts[1]), float(parts[2]), float(parts[3])]
                        self.vertices_3d_list[i].append([x, y, z])

    def step(self, json_file_path):
        json_data = None
        with open(json_file_path, "r") as file:
            json_data = json.load(file)
        
        # make a list of all objects contained in the json file
        points_2d_list = []
        
        for j in range(len(json_data)):
            points_2d = []
            num_of_good_points = 0 # if they have less than 6 keypoints with confidence > 0 then they are bad

            for i in range(len(json_data[j]["keypoints"]) // 3):    
                x_, y_, confidence = json_data[j]["keypoints"][i * 3: i * 3 + 3]
                x_, y_ = self.scene.pixel_to_NDC(x_, y_)

                points_2d.append({"point": [x_, y_], "confidence": confidence})

                if confidence > 0:
                    num_of_good_points += 1

            if num_of_good_points >= 6:
                points_2d_list.append({"points": points_2d, "bbox": json_data[j]["bbox"]})

        # for every existing car, see if there is one new one that is close
        matched_cars = []
        for car in self.cars:

            last_centerorid = Scene_manager.get_centroid(car.get_bbox())
            last_centerorid = self.scene.pixel_to_NDC(last_centerorid[0], last_centerorid[1])
            
            min_distance = np.inf
            min_index = -1

            for j in range(len(points_2d_list)):
                this_centeroid = Scene_manager.get_centroid(points_2d_list[j]["bbox"])
                this_centeroid = self.scene.pixel_to_NDC(this_centeroid[0], this_centeroid[1])

                distance = dis(this_centeroid, last_centerorid)

                if distance < min_distance:
                    min_distance = distance
                    min_index = j

            if min_distance <= self.max_distance_traveled:
                print("succesfully matched car")
                car.fit(points_2d=points_2d_list[min_index]["points"])
                matched_cars.append(car)

                points_2d_list.remove(points_2d_list[min_index])

        self.cars = matched_cars

        # for every remaining new car, create a new car (assume it just entered the frame)
        for points_2d in points_2d_list:
            car = PoseApproximation(vertices_3d_list=self.vertices_3d_list, projection_matrix=self.projection_matrix, obj_file_paths=self.obj_file_paths, bounding_box=points_2d["bbox"], framerate=self.frame_rate)
            car.fit(points_2d=points_2d["points"])
            self.cars.append(car)

    def visualize(self):
        for car in self.cars:
            car.visualize(self.scene.n * 1.1, self.scene.f)

    def render_frame(self):     
        self.scene.set_background(self.input_path + str(self.frame_number).rjust(4, "0") + ".jpg")
        self.scene.render_frame(self.output_path + str(self.frame_number).rjust(4, "0") + ".png")
        self.frame_number += 1

    def get_centroid(bbox):
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        return (cx, cy)        
