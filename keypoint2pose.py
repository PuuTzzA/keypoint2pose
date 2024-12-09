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

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>> clear all existing mesh objects in the scene and set the "cars" collection as the active one >>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

collections_to_keep = {"Collection", "arrow"} 
for obj in list(bpy.data.objects):
    keep_object = any(coll.name in collections_to_keep for coll in obj.users_collection)
    if not keep_object:
        bpy.data.objects.remove(obj, do_unlink=True)

bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children['cars']

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>> create Paths to the In- and Output files and initialize the scene_manager >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

current_dir = os.path.dirname(bpy.data.filepath)  # Blender's current working directory
car_models = ["volvo_v60", "ford_explorer", "nissan_altima"]
obj_file_paths = []
obj_keypoints_file_paths = []

for car_model in car_models:
    obj_file_paths.append(os.path.join(current_dir, "3d_model/" + car_model + "_wireframe.obj"))
    obj_keypoints_file_paths.append(os.path.join(current_dir, "3d_model/" + car_model + "_keypoints.obj"))

input_file_path = os.path.join(current_dir, "test/video1/video1_")
output_file_path = os.path.join(current_dir, "test/video1_out/video1_out_")

start_frame = 8
end_frame = 205 # inclusive

scene_manager = Scene_manager(obj_file_paths=obj_file_paths, keypoints_file_paths=obj_keypoints_file_paths, \
                              input_path=input_file_path, output_path=output_file_path, focal_length=52, sensor_width=36, \
                              frame_rate=30, start_index=start_frame)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>> iterate over all available frames and render them >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

for i in range(start_frame, end_frame + 1):
    scene_manager.step(json_file_path=input_file_path + str(i).rjust(4, "0") + ".json")
    scene_manager.visualize()
    scene_manager.render_frame() 

    print("new-frame-new-frame-new-frame-new-frame-new-frame-new-frame-new-frame-new-frame-new-frame-new-frame-new-frame")

print("-------------------------------------------------------------")   
print("-------------------------------------------------------------")   