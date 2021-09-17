# Jetson Object Security Project

## Project Ideia

The idea of this project is to develop a monitoring system for an object of interest through a Logitech C920E Business FullHD 1080p 960-001360 webcam  and a NVIDIA Jetson Xavier™ NX. The system should classify as high risk if someone gets within 1 meter of the object, medium risk between 1 and 1.25 meters and safe at a distance greater than 1.25 meters. We can imagine that this is a valuable or even dangerous object that you want to monitor.

## Project Structure

In the folder [DATA](https://drive.google.com/drive/folders/1fPqe5gvea7AtnlgAdFLNLum493Efg4Wu?usp=sharing), there is an informative presentation of the project, the YOLOv4 weights that should be placed in the "**model**" folder, dataset examples and runtime videos. These videos feature videos of Jetson's screen and the resulting videos from the monitoring.


## Project Set up

1. The screen, mouse and keyboards were used for calibration and visualization.
   - The first step was to set up jetson with JetPack following the instructions from the Jetson AI Fundamentals course (First Time Setup with JetPack).
     - The second step was to compile OpenCV with CUDA to fully access the GPU in inference time. I followed the script from this [github](https://github.com/mdegans/nano_build_opencv/blob/master/build_opencv.sh)
