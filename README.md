# Jetson Object Security Project

## Project Ideia

The idea of this project is to develop a monitoring system for an object of interest through a Logitech C920E Business FullHD 1080p 960-001360 webcam  and a NVIDIA Jetson Xavier™ NX. The system should classify as high risk if someone gets within 1 meter of the object, medium risk between 1 and 1.25 meters and safe at a distance greater than 1.25 meters. We can imagine that this is a valuable or even dangerous object that you want to monitor.

## Project Structure

In the folder [data](https://drive.google.com/drive/folders/1fPqe5gvea7AtnlgAdFLNLum493Efg4Wu?usp=sharing), there is an informative presentation of the project, the YOLOv4 weights that should be placed in the "**model**" folder, dataset examples and runtime videos. These videos feature videos of Jetson's screen and the resulting videos from the monitoring.

<p align="center">
 <img src="https://github.com/flaviobarbosaisi/Jetson-Obj-Security/blob/main/data/structure.jpeg">
</p>

## Project Set up

<p align="center">
  <img src="https://github.com/flaviobarbosaisi/Jetson-Obj-Security/blob/main/data/setup1.jpeg" width="256" height="256">
  <img src="https://github.com/flaviobarbosaisi/Jetson-Obj-Security/blob/main/data/setup2.jpeg" width="256" height="256">
</p>




   - The screen, mouse and keyboards were used for calibration and visualization.
   - The first step was to set up jetson with JetPack following the instructions from the Jetson AI Fundamentals course (First Time Setup with JetPack).
   - The second step was to compile OpenCV with CUDA to fully access the GPU in inference time. I followed the script from this [github](https://github.com/mdegans/nano_build_opencv/blob/master/build_opencv.sh)

## Dataset creation

The dataset was created with 10 videos of me walking around and moving the object of interest through the region of interest (ROI). As we are still in a pandemic period in my country, the class “person” were made only considering me, but this can be easily expanded with new videos and expansion of the dataset for better generalization of the "person" class. [Labelimg](https://github.com/tzutalin/labelImg) was used to create the labels.


![alt text](https://github.com/flaviobarbosaisi/Jetson-Obj-Security/blob/main/data/dataset.jpeg)

## Training

YOLOv4 training took place on a dgx A100. 827 images were used for training and 100 for testing. The training parameters and evaluations are presented below.

## Results and conclusion

The video results can be found [here](https://drive.google.com/drive/folders/19S-Pk4NGvOWUvMeMYiI6Trd-CuXnp9gJ), highlighting the GPU usage. The results proved to be efficient for the proposed application, as tt was possible to absorb an immense amount of knowledge about computer vision and machine learning.

## Usage

Download the yolo weights from [here](https://drive.google.com/file/d/1z_uqgPwsyQbfoNpk4vHIo8IXTZcSX9wr/view?usp=sharing) and place in the "**model**" folder. Modify the *FILE_PATH* variable to zero to access the usb webcam and *SUB_PATH* to the model directory. In case of testing with recorded video, modify *FILE_PATH* variable to the specific video path. There is a sample [here]().







