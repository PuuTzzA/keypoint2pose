import bpy
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import importlib
import source.scene as scene
import source.approximation as approximation
import source.algorithms as algorithms
import source.scene_manager as scene_manager
importlib.reload(scene)
importlib.reload(approximation)
importlib.reload(algorithms)
importlib.reload(scene_manager)
from source.scene import Scene
from source.algorithms import * 
from source.approximation import *
from source.scene_manager import Scene_manager 

# Clear all existing mesh objects in the scene
collections_to_keep = {"Collection", "arrow"}  # Replace with your collection names
for obj in list(bpy.data.objects):
    keep_object = any(coll.name in collections_to_keep for coll in obj.users_collection)
    if not keep_object:
        bpy.data.objects.remove(obj, do_unlink=True)

# Set this as the active collection
bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children['cars']

# Set the path to your OBJ file
current_dir = os.path.dirname(bpy.data.filepath)  # Blender's current working directory

car_models = ["volvo_v60", "ford_explorer", "nissan_altima"]
obj_file_paths = []
obj_keypoints_file_paths = []

for car_model in car_models:
    obj_file_paths.append(os.path.join(current_dir, "3d_model/" + car_model + "_wireframe.obj"))
    obj_keypoints_file_paths.append(os.path.join(current_dir, "3d_model/" + car_model + "_keypoints.obj"))

input_file_path = os.path.join(current_dir, "test/real_data_test_")
output_file_path = os.path.join(current_dir, "output/test_overlayed_")

scene_manager = Scene_manager(obj_file_paths=obj_file_paths, keypoints_file_paths=obj_keypoints_file_paths, \
                              input_path=input_file_path, output_path=output_file_path)

json_file_path = os.path.join(current_dir, "test/real_data_test_0001.jso  n")

scene_manager.step(json_file_path=json_file_path)
scene_manager.visualize()
scene_manager.render_frame() 

print("new frame new frame new frame new frame new frame new frame new frame")

json_file_path = os.path.join(current_dir, "test/real_data_test_0002.json")

scene_manager.step(json_file_path=json_file_path)
scene_manager.visualize()
scene_manager.render_frame() 


print("----------------------------------")   
print("----------------------------------")   