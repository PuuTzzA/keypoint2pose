# Keypoint2Pose


A python project to estimate the modelview matrix of a list of 2d key points to 3d vertices correspondences.


## Input


* **2d Key Points:** The 2d key points are given in a json file in this format:


    ```
    [
        {
            "keypoints" : [x1, y1, x2, y2, ...],
            "bbox": [x1, y1, x2, y2]
        },
        {
            "keypoints" : [x1, y1, x2, y2, ...],
            "bbox": [x1, y1, x2, y2]
        }
    ]
    ```


    This is a list of objects to approximate that exist in the current frame.


* **3d Vertex Data:** And the 3d model whose pose should be estimated is given as an .obj file. It is important that the order of vertices in the .obj file match the order of the corresponding key point in the json file. The scene manager takes a list of obj's and tries to find the best fitting one if the exact 3d model of the keypoints is not known.


* **Background Image:** Additionally there should also be an image file/sequence where the key points are taken from (this is only for visualizing the output, it has no functional purpose).


    The json files and according images should live in the same place, and have the same name, ending with "000" + #frame.


* **Camera Parameters:** The model also needs the camera's focal length and sensor size to accurately predict the object's pose. If you want to approximate the speed over multiple frames, the framerate is also needed.


[Blender's](https://www.blender.org/) python interface is used to create and visualize the pose estimation. To run the script, load into Blender and run it from there.


## Project Structure


To use it you can use the Scene_manager class to control all the logic.


```py
class Scene_manager():
    def __init__(self, obj_file_paths, keypoints_file_paths, input_path, output_path, max_distance=0.1, focal_length=30, frame_rate=30, start_index=1):
        pass

    def step(self, json_file_path):
        pass

    def visualize(self):
        pass

    def render_frame(self):
        pass

    def get_centroid(bbox):
        pass
```


The Scene_manager class manages a Scene and a List of PoseApproximation's. The scene is a representation of a blender scene and a poseApproximation is one detected object whose pose should be approximated.


```py
class Scene():
    def __init(self, resolution = (1920, 1080), n = 2, f = 30, focal_length = 30, sensor_width = 36):
        pass

    def setup_blender_scene(self):
        pass

    def get_projection_matrix(self):
        pass

    def create_view_frustum_and_background(self, l, r, t, b, n, f, fov, fov_vertical):
        pass

    def pixel_to_NDC(self, x, y):
        pass

    def set_background(self, path):
        pass

    def render_frame(self, output_path):
        pass

    def compute_frustum(focal_length, sensor_width, near, resolution):
        pass
```


```py
class PoseApproximation():
    def __init__(self, vertices_3d_list, projection_matrix, obj_file_paths, bounding_box, framerate=30, ransac_n=8, ransac_d=6):
        pass

    def fit(self, points_2d):
        pass

    def visualize(self, line_start, line_end):
        pass

    def get_bbox(self):
        pass

    def create_line(start, end, thickness=0.007):
        pass
```


In the ./source/file algorithms.py are the algorithms to approximate the pose (Direct Linear Transform) and the class RANSAC that is used to select the optimal keypoints.


```py
def direct_linear_transform(sample):
        pass


class RANSAC():
    def __init__(self, n, k, t, d, model, loss):
        pass

    def reset(self):
        pass

    def fit(self, data):
        pass
```
