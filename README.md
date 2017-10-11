## **List of Projects**  

**SLAM Demo: Extended Kalman Filter** [(project)](https://github.com/LukePhairatt/SLAM_DEMOS/tree/master/EKF_Slam)  
The project uses the offline data of motions and landmark measurements/observations (Lidar- range and bearing measurement) for the simulation. The work also demonstrates   
1. Full update   
2. Sparse matrix update  


**SLAM Demo: Particle Filter using Landmarks** [(project)](https://github.com/LukePhairatt/SLAM_DEMOS/tree/master/PF_FastSlam_Lmk)  
The project uses the offline data of motions and landmark measurements/observations (Lidar- range and bearing measurement)  
for the simulation. The work demonstrates: 
1. Landmark feature approach (cylinder landmarks) [(file)](https://github.com/LukePhairatt/SLAM_DEMOS/blob/master/PF_FastSlam_Lmk/src/pf_slam_lmk.py)   


**SLAM Demo: Particle Filter using Scan-matching** [(project)](https://github.com/LukePhairatt/SLAM_DEMOS/tree/master/PF_FastSlam_GridMap)  
The project uses the offline data of motions and landmark measurements/observations (Lidar- range and bearing measurement)  
for the simulation. The work demonstartes: 
1. Scan matching (Occupancy Grid Map) approach [(file)](https://github.com/LukePhairatt/SLAM_DEMOS/blob/master/PF_FastSlam_GridMap/src/pf_slam_mapmatching.py)  


**SLAM Demo: Online Graph**  
Online Graph SLAM for mapping.  

[RawLidarMotion](https://github.com/LukePhairatt/SLAM_DEMOS/tree/master/OnlineGraphSlam/RawLidar_Project)  
The experiment makes use the data from EKF/PF project on the implementation of Scalable Online Graph SLAM for landmarks mapping. I have implemented 2 models of the observation process: (range and bearing) and (distance x and distance y). The formula is based on a Probabilistic Robotics book (Thrun etc.) with 3DoF robot pose. 

[UdacityAI](https://github.com/LukePhairatt/SLAM_DEMOS/tree/master/OnlineGraphSlam/UdacityAI_Project)  
The work here provides the basic implementation on the graph update without the formula. It uses the generated measurements (dx,dy distance to a landmark) and motion command (dx,dy motion). The robot state is considering only the x,y position with no heading. The landmark measurement is a distance dx,dy from the current robot position. 

**Finding Lane** [(project)](https://github.com/LukePhairatt/SDC-FindingLaneLines-Project1)  
Video processing to detect lane lines.

Image processing::  
RGB threshlding, Gaussian Blur, ROI mask, Hough transform, Polynomial fitting  

**Traffic Sign Classification** [(project)](https://github.com/LukePhairatt/SDC-TrafficSignClassifier-Project2)  
Image processing for a traffic signs recognition using Convolutional Neural Network (TensorFlow).  

Pipline processing::  
Normalised to gray scale, Data balance (Image augmentation), LeNet model, Train and save Network, Prediction   

**Behavioral Cloning of Self-Driving Cars** [(project)](https://github.com/LukePhairatt/SDC-Behaviral-Cloning-Project3)  
Build and train a Deep Neural Network (modified Nvidia) to predict the steering angle.  
The image data and corresponding inputs (e.g. steering angle) are generated from recording the driving around the tracks in the simulator.  

Pipline processing:: 
Generate the data (left,centre,right image),Resize images,Data generation (Balance data), Train the network, Run simulator. 

[(challenge)](https://github.com/LukePhairatt/SDC-Behaviral-Cloning-Project3/tree/master/track2) Track 2 is more challenging with twist corners going up and down hills.

[(my exploration)](https://github.com/LukePhairatt/SDC-Behaviral-Cloning-Project3/tree/master/track1_racing) Variable speed control in addition to only a steering control

**Advance Lane Detection** [(project)](https://github.com/LukePhairatt/SDC-Advanced-Lane-Finding-Project-4)  
Simiilar to the Finding Lane Project, this project implements a more robust version in detection with a camera calibration  
to undistorted images and warps them to the real world space to compute radius of curvature and centre lane usefuly for othercar control systems. 

Pipline processing::
Camera calibration,Undistorted image, Binary image (Sobel,RGB, HLS thresholding),
Extract binary lane lines(window search and histogram peak), warped lane lines, fit curves and compute centre

**Vehicle Detection and Tracking** [(project)](https://github.com/LukePhairatt/SDC-VehicleTracking-Project-5)  
Build the video processing pipline to detect and track cars in the image frame.  
The work features Color spatial, Color histogram and HOG descriptor for training the SVM classifier.  
Note: Input images can be ['HSV','LUV','RGB','YCrCb'] color space of 3 channels.

Pipline processing::  
Balance data, Extract spatial, histogram, hog features, Train SVM, Save model, Prediction

**Kalman and Extended Kalman Filter** [(project)](https://github.com/LukePhairatt/SDC-EKFLidarRadarFusion-Project1T2)  
This is a Kalman Filter for an object tracking application. The obseravtions are the measurement of the moving location in time  
given by Lidar(x,y) and Radar(range, angle, range changing rate).

**Unsecented Kalman Filter** [(project)](https://github.com/LukePhairatt/SDC-UKFLidarRadarFusion-Project2T2-)  
This is a Unsecented Kalman Filter for an object tracking application. The obseravtions are the measurement of the moving location in time given by Lidar(x,y) and Radar(range, angle, range changing rate).

[(challenge)](https://github.com/LukePhairatt/SDC-UKFLidarRadarFusion-Project2T2-/tree/master/bonus_challenge/CarND-Catch-Run-Away-Car-UKF) Catching the moving object using my look ahead predictor. 

**Particle Filter** [(project)](https://github.com/LukePhairatt/SDC-ParticleFilterNavigation-Project3T2)  
Mobile robot navigation project based on Markov model assumption. The state correction makes use of the landmark measurement in the given map to compute likelihood for a resampling process. 

**PID Steering Control** [(project)](https://github.com/LukePhairatt/SDC-PID-Project4T2)  
Implementation of a PID controller to make a car follow the path indicated by the cross-track error given by the simulator.  
The work includes the time dependance Kd term in order to run on different PC and configurations.

**Model Predictive Steering Control** [(project)](https://github.com/LukePhairatt/SDC-MPC-Project5T2)  
The project implements MPC to control the steering and throttle output to follow the planned path given by the way points.  
In addition, latency in sending these control commands (e.g. 100 ms) has been added to simulate the real world response.
 
This project makes use of the non-linear solver to the optimisation problem of predicting the control outputs within the given constrains. The constraints are given by cross track error, heading error, control costs, control limits, reference speed for example 

**Behavior Planning and Trajectory Generation**[(project)](https://github.com/LukePhairatt/SDC-PathPlanning-Project1-T3)  
The project presents the implementation of behavior planning and spline line trajectory generation for autonomous driving of the self-driving cars on the highway. Behavior of the car is controlled by the associate cost functions to reach the setting goal point.    

**Fully Convolutional Neural Network (FCN) for Image Segmentation**[(project)](https://github.com/LukePhairatt/SDC-Semantic-Segmentation-FCN-Project2-T3)  
The project implements FCN for the detection of 'Road' and 'Not Road' pixels in the images. FCN utilises VGG-16 model with encoding and decoding with skip layers on the pre-trained VGG model.


**ROS Robot Development** [(project)](https://github.com/LukePhairatt/ROSRPiArduinoProject)  
The development of a differential drive robot base with a depth RGB camera running ROS. The robot motor drive is controlled by Arduino. ROS Arduino serial interface, navigation stack, and vision modules are running on RPi2.  

**FFT Robust Gaussian Regression Filter (RGR)** [(project)](https://github.com/LukePhairatt/RobustGaussianRegression)  
Algorithm for micro defect detections(~25 um or more). The core idea is to use RGR to re-engineer the nominal surface for thresholding.The binary surface residuals can then be detected using convention image processing techniques.   


**Line Laser Scanner Hand-Eye Calibration**[(project)](https://github.com/LukePhairatt/LaserHandEyeCalibration)  
MATLAB implementation of computing 6DoF pose of a line laser scanner frame w.r.t a robot tool frame.   









