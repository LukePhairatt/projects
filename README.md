## **List of Projects**  

**SLAM Demo: Extended Kalman Filter** [(project)](https://github.com/LukePhairatt/SLAM_DEMOS/tree/master/EKF_Slam)  
The project uses the offline data of motions and landmark measurements/observations (Lidar- range and bearing measurement) for the simulation. The work also demonstrates   
1. full update   
2. sparse matrix update  


**SLAM Demo: Particle Filter**[(project)](https://github.com/LukePhairatt/SLAM_DEMOS/tree/master/PF_FastSlam_GridMap)  
The project uses the offline data of motions and landmark measurements/observations (Lidar- range and bearing measurement)  
for the simulation. The work demonstartes: 
1. Feature based (cylinder landmarks)  
2. Scan matching (Occupancy Grid Map) based navigation.  


**SLAM Demo: Online Graph**  
Online Graph SLAM for mapping.  

[RawLidarMotion]()  
The experiment makes use the data from EKF/PF project on the implementation of Scalable Online Graph SLAM for landmarks mapping.  
I have implemented 2 models of observation: (range and bearing) and (distance x and distance y). The formula is based on a Probabilistic Robotics book (Thrun etc.) for 3DoF robot pose. 

[UdacityAI]()  
The work here is simpler. It uses the generated measurements (distance to a landmark) and motion command. The robot state is x,y position  
with no heading. The landmark measurement is a distance x,y from the current robot position. 

**Finding Lane** [project]()  
Video processing to detect lane lines.

Image processing::  
RGB threshlding, Gaussian Blur, ROI mask, Hough transform, Polynomial fitting  

**Traffic Sign Classification** [project]()  
Image processing for a traffic signs recognition using Convolutional Neural Network (TensorFlow).  

Pipline processing::  
Normalised to gray scale, Data balance (Image augmentation), LeNet model, Train and save Network, Prediction   

**Behavioral Cloning of Self-Driving Cars** [project]()  
Build and train a Deep Neural Network (modified Nvidia) to predict the steering angle.  
The image data and corresponding inputs (e.g. steering angle) are generated from recording the driving around the tracks in the simulator.  

Pipline processing:: 
Generate the data (left,centre,right image),Resize images,Data generation (Balance data), Train the network, Run simulator. 

[challenge]() Track 2 is more challenging with twist corners going up and down hills.
[my exploration]() Variable speed control in addition to only a steering control

**Advance Lane Detection** [project]()  
Simiilar to the Finding Lane Project, this project implements a more robust version in detection with a camera calibration  
to undistorted images and warps them to the real world space to compute radius of curvature and centre lane usefuly for othercar control systems. 

Pipline processing::
Camera calibration,Undistorted image, Binary image (Sobel,RGB, HLS thresholding),
Extract binary lane lines(window search and histogram peak), warped lane lines, fit curves and compute centre

**Vehicle Detection and Tracking** [project]()  
Build the video processing pipline to detect and track cars in the image frame.  
The work features Color spatial, Color histogram and HOG descriptor for training the SVM classifier.  
Note: Input images can be ['HSV','LUV','RGB','YCrCb'] color space of 3 channels.

Pipline processing::  
Balance data, Extract spatial, histogram, hog features, Train SVM, Save model, Prediction

**Kalman and Extended Kalman Filter** [project]()  
This is a Kalman Filter for an object tracking application. The obseravtions are the measurement of the moving location in time  
given by Lidar(x,y) and Radar(range, angle, range changing rate).

**Unsecented Kalman Filter** [project]()  
This is a Unsecented Kalman Filter for an object tracking application. The obseravtions are the measurement of the moving location in time given by Lidar(x,y) and Radar(range, angle, range changing rate).

[challenge]() Catching the moving object using my look ahead predictor. 

**Particle Filter** [project]()  
Mobile robot navigation project based on Markov model assumption. The state correction makes use of the landmark measurement in the given map to compute likelihood for a resampling process. 

**PID Steering Control** [project]()  
Implementation of a PID controller to make a car follow the path indicated by the cross-track error given by the simulator.  
The work includes the time dependance Kd term in order to run on different PC and configurations.

**Model Predictive Steering Control** [project]()  
The project implements MPC to control the steering and throttle output to follow the planned path given by the way points.  
In addition, latency in sending these control commands (e.g. 100 ms) has been added to simulate the real world response.
 
This project makes use of the non-linear solver to the optimisation problem of predicting the control outputs within the given constrains. The constraints are given by cross track error, heading error, control costs, control limits, reference speed for example 

**ROS Robot Development** [project]()  
The development of a differential drive robot base with a depth RGB camera running ROS. The robot motor drive is controlled by  
Arduino. ROS Arduino serial interface, navigation stack, and vision modules are running on RPi2.

**FFT Robust Gaussian Regression Filter (RGR)** [project]()  
Algorithm for micro defect detections. The core idea is to use RGR to re-engineer the nominal surface form for thresholding.  
The binary surface residuals can then be detected using convention image processing techniques. 

