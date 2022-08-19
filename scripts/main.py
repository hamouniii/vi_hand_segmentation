from asyncore import read
from importlib.resources import path
import queue
from turtle import left
from grpc import server
from importlib_metadata import re
from sklearn.feature_extraction import img_to_graph
from sympy import true
import rospy
from std_msgs.msg import String
import actionlib
import actionlib_tutorials

from sensor_msgs.msg import Image, CameraInfo, Imu
from cv_bridge import CvBridge, CvBridgeError


import message_filters
import sys
from time import time
import timeit


import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

import random
import torch.nn.functional as F
# from torchmetrics import IoU
import matplotlib
import matplotlib.pyplot as plt
import imutils
import threading, queue
from csv import writer, reader

import pyrealsense2 as rs
import tf
# import filterpy
# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise

# from user_defined_msgs.msg import HandDetectionInferenceAction, HandDetectionInferenceFeedback, HandDetectionInferenceResult, HandDetectionInferenceGoal




def normalize_tensor(tensor, mean, std):

    for t in tensor:
        t.sub_(mean).div_(std)

    return tensor





class HandSegmentation:
    def __init__(self, model):
        self.cv_bridge = CvBridge()

        self.color_sub = rospy.Subscriber("/d435i/color/image_raw", Image, self.color_cb)
        self.color_cam_info_sub = rospy.Subscriber("/d435i/color/camera_info", CameraInfo, self.color_cam_info_cb)
        self.device = "cpu"
        self.color_projection_matrix = None
        self.listener = tf.TransformListener()

        self.hand_detection_model = model
        self.csv_file = "hand_segmentation.csv"
        with open(self.csv_file, 'a+', newline='') as write_obj:
                csv_writer = writer(write_obj)
                to_write = ["seq", 
                "lu", "lv", "lu_p", "lv_p", "lx", "ly", "lz", "lx_p", "ly_p", "lz_p",
                "ru", "rv", "ru_p", "rv_p", "rx", "ry", "rz", "rx_p", "ry_p", "rz_p"] 
                csv_writer.writerow(to_write)
                write_obj.close()




    def color_cam_info_cb(self, msg):
        self.color_projection_matrix = np.array(msg.P).reshape((3,4))
        K = msg.K

        self.color_params = rs.intrinsics()
        self.color_params.width = msg.width
        self.color_params.height = msg.height
        self.color_params.model = rs.distortion.none

        self.color_params.ppx = K[2]
        self.color_params.ppy = K[5]
        self.color_params.fx = K[0]
        self.color_params.fy = K[4]


    def color_cb(self, msg):
        color_seq_number = msg.header.seq
        (l_trans, l_quat) = self.listener.lookupTransform('/d435i_color_optical_frame', '/Seg_LeftHand', rospy.Time(0))
        (r_trans, r_quat) = self.listener.lookupTransform('/d435i_color_optical_frame', '/Seg_RightHand', rospy.Time(0))
        depth_msg = rospy.wait_for_message("/d435i/aligned_depth_to_color/image_raw", Image, timeout=1)
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # device = "cpu"
        
        img_r = cv_image.copy()
        H, W = cv_image.shape[:2]
        preprocessed_img = self.hand_detection_model.preprocess(cv_image)

        with torch.no_grad():
            pred = self.hand_detection_model.model(preprocessed_img.to(self.device))

            if isinstance(pred, tuple) or isinstance(pred, list):
                pred = pred[0]
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)[0]
            
            pred = F.softmax(pred, dim=0)
            
            mask = torch.argmax(pred, dim=0).cpu().numpy()

            img_r[mask == 1] = [80, 80, 80] # right
            img_r[mask == 2] = [160, 160, 160]

            left_pixels  = np.where(img_r == [80, 80, 80])
            right_pixels = np.where(img_r == [160, 160, 160])
            
            unique_coordinates_left  = np.asarray(left_pixels[0:2]).T
            unique_coordinates_right = np.asarray(right_pixels[0:2]).T
            
            mean_left  = np.mean(unique_coordinates_left, axis = 0)
            mean_right = np.mean(unique_coordinates_right, axis = 0)

            lu_p = int(mean_left[0])
            lv_p = int(mean_left[1])
            ru_p = int(mean_right[0])
            rv_p = int(mean_right[1])

            lx = l_trans[0]; ly = l_trans[1]; lz = l_trans[2]
            rx = r_trans[0]; ry = r_trans[1]; rz = r_trans[2]

            l_p = rs.rs2_deproject_pixel_to_point(self.color_params, [lu_p, lv_p], depth_img[lv_p, lu_p])
            r_p = rs.rs2_deproject_pixel_to_point(self.color_params, [ru_p, rv_p], depth_img[rv_p, ru_p])

            l_p = [x/1000 for x in l_p]
            lx_p = l_p[0]; ly_p = l_p[1]; lz_p = l_p[2]
            r_p = [x/1000 for x in r_p]
            rx_p = r_p[0]; ry_p = r_p[1]; rz_p = r_p[2]


            l_4d = l_trans; r_4d = r_trans
            l_4d.append(1); r_4d.append(1)

            # l_4d = [lx, ly, lz, 1.0]
            # r_4d = [rx, ry, rz, 1.0]
            lZ = self.color_projection_matrix.dot(np.array(l_4d).reshape((4,1)))
            rZ = self.color_projection_matrix.dot(np.array(r_4d).reshape((4,1)))
            
            print(lZ)
            lu = int(lZ[0]/lz[2])
            lv = int(lZ[1]/lz[2])
            ru = int(rZ[0]/rz[2])
            rv = int(rZ[1]/rz[2])


            to_write = [color_seq_number, lu, lv, lu_p, lv_p, lx, ly, lz, lx_p, ly_p, lz_p,
                ru, rv, ru_p, rv_p, rx, ry, rz, rx_p, ry_p, rz_p]
            cv.circle(img_r, (int(mean_right[1]), int(mean_right[0])), 2,(255, 280, 255), 2)

            
            cv.imshow("Image window wait", img_r)
            cv.waitKey(3)
         
        


# def preprocess2(img, size=(512, 288), grayscale=False, input_edge=False): # size=(512, 288)

#     if grayscale:
#         img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#     if input_edge:
#         edge = cv.Canny(img.astype(np.uint8), 25, 100)
#         img = np.stack((img, edge), -1)
  
#     img = cv.resize(img, size)
   
#     if not grayscale:

#         img = img / 255.0
#         IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
#         IMG_VARS = np.array((0.229, 0.224, 0.225), dtype=np.float32)

#         img -= IMG_MEAN
#         img /= IMG_VARS

#         img = np.transpose(img, (2, 0, 1))

#     if input_edge:
#         img = np.transpose(img, (2, 0, 1))

#     img = torch.from_numpy(img).float()
#     img = img.unsqueeze(0)

#     if grayscale:
#         img = normalize_tensor(img, 128.0, 256.0)

#     if grayscale and not input_edge:
#         img = img.unsqueeze(0)

#     return img

# def setup_model(model_name, weights_path=None, grayscale=False, input_edge=False):

#     if grayscale:
#         in_channels = 1
#     if input_edge:
#         in_channels = 2
#     else:
#         in_channels = 3

#     model = SEG_MODELS_REGISTRY.get(model_name)(in_channels=in_channels, n_classes=3)
#     # model = BiSeNet()(in_channels=inBiSeNet_channels, n_classes=3)

#     # model = torch.load(weights_path, map_location=torch.device('cpu'))
#     print("MODEEEEEEL FOUNDDDDD")

#     if weights_path is not None:
#         model.load_state_dict(
#             torch.load(weights_path, map_location=torch.device("cpu")),
#         )

#     model.eval()

#     return model


# def warmup(model, device, inp_size=512, in_channels = 2):
#     print('--------------  Warm Up  ----------')
#     inp = torch.rand(1, in_channels, inp_size[0], inp_size[1]).to(device)
#     model = model.to(torch.device(device))
#     model(inp)

# def run_demo2(image):
#     print("Demo2")
#     # cv.imshow("Image window25", image)
#     # cv.waitKey(0)




# def run_demo(cubic_size,model, weights_path, grayscale, input_edge, ic, out_dir=None, device="cpu", size=(512, 288)):

#     imageHeight, imageWidth, fps = 480, 640, 60
#     # Camera  = IntelRealSense_camera(imageHeight, imageWidth, fps)
#     Camera = 1
#     print(ic.d435i_image_sub)
    
                        
#     pipeline, depth_scale, depth_intrinsics, pc, rgb_intrinsics  = Camera.init_device() 
#     fx , ux, fy , uy, idx = rgb_intrinsics.fx , rgb_intrinsics.ppx , rgb_intrinsics.fy , rgb_intrinsics.ppy , 0
#     #model = model.to(torch.device(device))
    
#     model = model.eval()

#     while True:

#         depth, image, clipping_distance, depth_frame, color = Camera.read_frame_from_device(pipeline, depth_scale) 
    
#         img_r = image.copy()
#         depth_image = depth.copy()
        
#         H, W = image.shape[:2]
#         img = preprocess2(image, size=size, grayscale=grayscale, input_edge=input_edge)
#         with torch.no_grad():
#             pred = model(img.to(device))
            
#             if isinstance(pred, tuple) or isinstance(pred, list):
#                 pred = pred[0]
#             pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)[0]
            
#             pred = F.softmax(pred, dim=0)
            
#             mask = torch.argmax(pred, dim=0).cpu().numpy()
#             img_r[mask == 1] = [127, 127, 255] # right
#             img_r[mask == 2] = [25, 120, 127]
#             # img_r[mask == 2] = [125, 125, 125]

#             left_pixels = np.where(img_r == [255, 120, 127])
#             unique_coordinates_left = np.asarray(left_pixels[0:2]).T
#             mean_left = np.mean(unique_coordinates_left, axis = 0)
#             cv.circle(img_r, (int(mean_left[1]), int(mean_left[0])), 2,(255, 280, 255), 2)

                  
#             cv.imshow('mask_mean_point', img_r)
            
#             if cv.waitKey(1) & 0xFF == ord("q"):
#                 break

    
#     Camera.stop_device(pipeline)



class BiSeNetModel:
    def __init__(self, model_name, weights_path=None, grayscale=True, input_edge=True):#setup
        if grayscale:
            self.in_channels = 1
        if input_edge:
            self.in_channels = 2
        else:
            self.in_channels = 3
    
        self.model = SEG_MODELS_REGISTRY.get(model_name)(in_channels=self.in_channels, n_classes=3)
        # model = BiSeNet()(in_channels=inBiSeNet_channels, n_classes=3)
    
        # model = torch.load(weights_path, map_location=torch.device('cpu'))
    
        if weights_path is not None:
            self.model.load_state_dict(
                torch.load(weights_path, map_location=torch.device("cpu")),
            )
    
        self.model.eval()
        self.n = 0
        self.ca = 0
        self.last = timeit.default_timer()
        # return self.model

    def warmup(self, device, inp_size=512, in_channels=2):
        print('--------------  Warm Up  ----------')
        inp = torch.rand(1, in_channels, inp_size[0], inp_size[1]).to(device)
        self.model = self.model.to(torch.device(device))
        self.model(inp)
    
    

    def infer(self, img):
      
        device = "cpu"
        img_r = img.copy()
        H, W = img.shape[:2]

        processed_img = self.preprocess(img)
        with torch.no_grad():
            
            start = timeit.default_timer()
            pred = self.model(processed_img.to(device))
            end  = timeit.default_timer()
            self.ca = (self.ca * self.n + (end - start))/(self.n+1)
            self.n = self.n+1
            # print("continous avg: {}".format(self.ca))
            if isinstance(pred, tuple) or isinstance(pred, list):
                pred = pred[0]

            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)[0]
            
            pred = F.softmax(pred, dim=0)
            
            mask = torch.argmax(pred, dim=0).cpu().numpy()

            print(mask.shape)
            print(mask)
            # img_r[mask == 1] = [127, 127, 255] # right
            # img_r[mask == 2] = [25, 120, 127]
            img_r[mask == 1] = [125, 0, 0]
            img_r[mask == 2] = [0, 0, 0]

            # left_pixels = np.where(img_r == [255, 120, 127])
            # unique_coordinates_left = np.asarray(left_pixels[0:2]).T
            # mean_left = np.mean(unique_coordinates_left, axis = 0)
            # cv.circle(img_r, (int(mean_left[1]), int(mean_left[0])), 2,(255, 280, 255), 2)

        cv.imshow("Image window", img_r)
        cv.waitKey(3)

    def preprocess(self, img, size= (512,256), grayscale=True, input_edge=True):
        if grayscale:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if input_edge:
            edge = cv.Canny(img.astype(np.uint8), 25, 100)
            img = np.stack((img, edge), -1)
    
        img = cv.resize(img, size)
    
        if not grayscale:

            img = img / 255.0
            IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
            IMG_VARS = np.array((0.229, 0.224, 0.225), dtype=np.float32)
    
            img -= IMG_MEAN
            img /= IMG_VARS
    
            img = np.transpose(img, (2, 0, 1))

        if input_edge:
            img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)

        if grayscale:
            img = normalize_tensor(img, 128.0, 256.0)

        if grayscale and not input_edge:
            img = img.unsqueeze(0)

        return img


# def init_kf(init_x):
#     kf = KalmanFilter(dim_x=4, dim_z=2)
#     dt = 0
#     kf.F = np.array([[1, dt, 0, 0]
#                     ,[0, 1,  0, 0]
#                     ,[0, 0,  1, dt]
#                     ,[0, 0,  0, 1]])
#     kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.1)
#     return kf
# def mykf():
#     kf = init_kf(init = np.array([0,0,0,0]))

#     pass

# def cal_time_difference(start, end):
#     pass



def wait_infer(model, ic):
    device = "cpu"
    data = None
    first = True
    kf_r = None
    kf_l = None
    while True:
        while data is None:
            try:
                data = rospy.wait_for_message("/d435i/color/image_raw", Image, timeout=5)
                if(first):
                    first = False
                    last_stamp = data.header.stamp

                print(last_stamp)
                
            except:
                print("exeption")

        cv_image = ic.ros_to_cv(data)

        img_r = cv_image.copy()
        H, W = cv_image.shape[:2]
        preprocessed_img = model.preprocess(cv_image)

        with torch.no_grad():
            pred = model.model(preprocessed_img.to(device))

            if isinstance(pred, tuple) or isinstance(pred, list):
                pred = pred[0]
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)[0]
            
            pred = F.softmax(pred, dim=0)
            
            mask = torch.argmax(pred, dim=0).cpu().numpy()

            # img_r[mask == 1] = [127, 127, 255] # right
            # img_r[mask == 2] = [25, 120, 127]


            img_r[mask == 1] = [80, 80, 80] # right
            img_r[mask == 2] = [160, 160, 160]


            # left_pixels = np.where(img_r == [255, 120, 127])
            left_pixels  = np.where(img_r == [80, 80, 80])
            right_pixels = np.where(img_r == [160, 160, 160])
            unique_coordinates_left  = np.asarray(left_pixels[0:2]).T
            unique_coordinates_right = np.asarray(right_pixels[0:2]).T
            # print(len(unique_coordinates_left))
            mean_left  = np.mean(unique_coordinates_left, axis = 0)
            mean_right = np.mean(unique_coordinates_right, axis = 0)


            cv.circle(img_r, (int(mean_left[1]), int(mean_left[0])), 2,(255, 280, 255), 2)
            cv.circle(img_r, (int(mean_right[1]), int(mean_right[0])), 2,(255, 280, 255), 2)


            cv.imshow("Image window wait", img_r)
            cv.waitKey(3)

        data = None
    

    
    


class CameraParams:
    def __init__(self):
        self.d435i_cam_info_sub = rospy.Subscriber("/d435i/color/camera_info", CameraInfo, self.d435i_cam_info_callback)
        print("Cam Info initialized!")

    def d435i_cam_info_callback(self, data):
        K = data.K
        self.fx  = K[0]
        self.fy  = K[4]
        self.ux  = K[2]
        self.uy  = K[5]
        self.idx = K[8]
        self.d435i_cam_info_sub.unregister()



class ImageConverter:
  
    def __init__(self):
        # self.image_pub = rospy.Publisher("image_topic_2",Image)
        # self.my_bisenet = InferDemo("BiSeNet")
        # self.my_bisenet.warmup("cpu", inp_size=(512, 256))

        self.bridge = CvBridge()
        # self.d435i_image_sub = rospy.Subscriber("/d435i/color/image_raw", Image, self.d435i_image_callback, queue_size=1)
        self.last = timeit.default_timer()
        print("image_converter initialized!")

  
    # def d435i_image_callback(self, data):
    #     now = timeit.default_timer()
    #     print(now-self.last)
    #     self.last = now
    #     cv_img = self.ros_to_cv(data)
  
    #     self.my_bisenet.infer(cv_img)

    #     cv.imshow("Image window CB", cv_img)
    #     cv.waitKey(3)
  
        # try:
        #     self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        # except CvBridgeError as e:
        #     print(e)
  
    def cv_to_ros(self):
        pass
    

    def ros_to_cv(self, ros_img_in):
        try:
            # cv_image = self.bridge.imgmsg_to_cv2(ros_img_in, "bgr8")
            cv_image = self.bridge.imgmsg_to_cv2(ros_img_in, 'bgr8')

        except CvBridgeError as e:
            print(e)
        return cv_image


class HandDetectionActionServer():
    def __init__(self, model):
        self.hand_detection_model = model
        self.server = actionlib.SimpleActionServer("/HandDetection", HandDetectionInferenceAction, execute_cb=self.hand_detection_requested_cb, auto_start=False)
        self.bridge = CvBridge()
        print("HandDetection Action Server Initialized...")


    def start(self):
        self.server.start()


    def hand_detection_requested_cb(self, msg):

        device = "cpu"
        cv_image = None

        if msg.img == None:
            print("reading msg directly from rosbag")
            while cv_image is None:
                try:
                    data = rospy.wait_for_message("/d435i/color/image_raw", Image, timeout=5)
                    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                except:
                    print("Exception in hand detection requested!!!!")

        else:
            cv_image = self.bridge.imgmsg_to_cv2(msg.img, "bgr8")


        img_r = cv_image.copy()
        H, W = cv_image.shape[:2]
        preprocessed_img = self.hand_detection_model.preprocess(cv_image)

        with torch.no_grad():
            pred = self.hand_detection_model.model(preprocessed_img.to(device))

            if isinstance(pred, tuple) or isinstance(pred, list):
                pred = pred[0]
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)[0]
            
            pred = F.softmax(pred, dim=0)
            
            mask = torch.argmax(pred, dim=0).cpu().numpy()

            img_r[mask == 1] = [80, 80, 80] # right
            img_r[mask == 2] = [160, 160, 160]


            left_pixels  = np.where(img_r == [80, 80, 80])
            right_pixels = np.where(img_r == [160, 160, 160])
            unique_coordinates_left  = np.asarray(left_pixels[0:2]).T
            unique_coordinates_right = np.asarray(right_pixels[0:2]).T
            
            mean_left  = np.mean(unique_coordinates_left, axis = 0)
            mean_right = np.mean(unique_coordinates_right, axis = 0)

            # print(mean_left)
            # print(mean_right)
            res = HandDetectionInferenceResult()

            res.xl = mean_left[0]
            res.yl = mean_left[1]
            res.xr = mean_right[0]
            res.yr = mean_right[1]
            # print(res)
            self.server.set_succeeded(result=res)


            #################
            cv.circle(img_r, (int(mean_right[1]), int(mean_right[0])), 2,(255, 280, 255), 2)

            cv.imshow("Image window wait", img_r)
            cv.waitKey(3)
            ################




# class Counter():
#     def __init__(self):
#         self.server = actionlib.SimpleActionServer("name", HandDetectionInferenceAction, execute_cb=self.execute_cb, auto_start=False)
#         self.sofar = 0
#         self.bridge = CvBridge()
#         self.server.start()
#         print("counter initialized")
#     def execute_cb(self, msg):
#         data = None
#         while data is None:
#             try:
#                 data = rospy.wait_for_message("/d435i/color/image_raw", Image, timeout=5)
                
#             except:
#                 print("exeption")
#         print(data)
#         cv_img = self.bridge.imgmsg_to_cv2(data, 'bgr8')

#         cv.imshow("Image window wait", cv_img)
#         cv.waitKey(3)
#         print("End of cb")
#         self.server.set_succeeded(1710)
#         print("Success")

#         # print("Start JOBBBBBBBBBBBBBBBBBB")
#         # print(type(data.max_number))
#         # for i in range(data.max_number):
#         #     i = i + 1
#         #     self.sofar += 1
#         # print("FInished Jobbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        



def main(args):
    rospy.init_node('image_converter', anonymous=True)
    # image_converter = ImageConverter()
    # _ = CameraParams()

    path_to_model = "models/Segmentation_weights/Segmentation_epochs/model_weights/grayscale/run2/BiSeNet.pth"

    model = BiSeNetModel("BiSeNet", weights_path=path_to_model)
    model.warmup("cpu", inp_size=(512, 256))

    hand_segmentation = HandSegmentation(model)

    # hand_detection = HandDetectionActionServer(model)
    # hand_detection.start()
    # wait_infer(model, image_converter)


    model_name = "BiSeNet"
    path_to_model = "models/Segmentation_weights/Segmentation_epochs/model_weights/grayscale/run2/BiSeNet.pth"
    grayscale = True
    input_edge = True

    device = 'cpu'
    size = (512, 256)
    cubic_size = 250
    out_dir = ""



    # model = setup_model(args.model, args.model_weights, args.grayscale, args.input_edge)
    # model = setup_model(model_    name, path_to_model, grayscale, input_edge)
    # warmup(model, args.device, size)
    # warmup(model, device, size)
    # run_demo(args.cubic_size, model, args.model_weights, args.grayscale, args.input_edge, args.out_dir, args.device, size)
    # run_demo(cubic_size, model, path_to_model, grayscale, input_edge, image_converter, out_dir, device, size)



    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")

    cv.destroyAllWindows()







#registry.py
"""
Adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
"""

from torch.optim import SGD, Adadelta, Adagrad, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


class Registry:
    def __init__(self, name):

        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):

        assert (
            name not in self._obj_map
        ), f"An object named '{name}' was already registered in '{self._name}' registry!"

        self._obj_map[name] = obj

    def register(self, obj=None, name=None):

        if obj is None:

            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        if name is None:
            name = obj.__name__

        self._do_register(name, obj)

    def get(self, name):

        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry!"
            )

        return ret

    def get_list(self):
        return list(self._obj_map.keys())

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())


optimizers = Registry("optimizers")
schedulers = Registry("schedulers")


optimizers.register(SGD, "SGD")
optimizers.register(Adam, "Adam")
optimizers.register(AdamW, "AdamW")
optimizers.register(Adagrad, "Adagrad")
optimizers.register(Adadelta, "Adadelta")
optimizers.register(RMSprop, "RMSprop")

schedulers.register(CosineAnnealingLR, "CosineAnnealingLR")
schedulers.register(CosineAnnealingWarmRestarts, "CosineAnnealingWarmRestarts")
schedulers.register(CyclicLR, "CyclicLR")
schedulers.register(MultiStepLR, "MultiStepLR")
schedulers.register(ReduceLROnPlateau, "ReduceLROnPlateau")
schedulers.register(StepLR, "StepLR")
schedulers.register(OneCycleLR, "OneCycleLR")
#registry.py end







SEG_MODELS_REGISTRY = Registry("SEGMENTATION_MODELS")







def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)

        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):

    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))

    return nn.Sequential(*layers)


class ResNet18(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32

        return feat8, feat16, feat32

    def get_params(self):

        wd_params, nowd_params = [], []

        for name, module in self.named_modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)

            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())

        return wd_params, nowd_params


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class UpSample(nn.Module):
    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()

        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):

        feat = self.proj(x)
        feat = self.up(feat)

        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()

        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(
            scale_factor=up_factor, mode="bilinear", align_corners=False
        )
        self.init_weight()

    def forward(self, x):

        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)

        return x

    def init_weight(self):

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):

        wd_params, nowd_params = [], []

        for name, module in self.named_modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)

            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())

        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()

        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):

        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)

        return out

    def init_weight(self):

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, in_channels=3):
        super(ContextPath, self).__init__()

        self.resnet = ResNet18(in_channels=in_channels)
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.0)
        self.up16 = nn.Upsample(scale_factor=2.0)

        self.init_weight()

    def forward(self, x):

        feat8, feat16, feat32 = self.resnet(x)

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.up32(feat32_sum)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up  # x8, x16

    def init_weight(self):

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):

        wd_params, nowd_params = [], []

        for name, module in self.named_modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)

            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())

        return wd_params, nowd_params


class SpatialPath(nn.Module):
    def __init__(self, in_channels=3):
        super(SpatialPath, self).__init__()

        self.conv1 = ConvBNReLU(in_channels, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):

        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)

        return feat

    def init_weight(self):

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):

        wd_params, nowd_params = [], []

        for name, module in self.named_modules():

            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)

            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())

        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()

        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)

        self.conv = nn.Conv2d(
            out_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, fsp, fcp):

        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = atten.sigmoid()
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat

        return feat_out

    def init_weight(self):

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):

        wd_params, nowd_params = [], []

        for name, module in self.named_modules():

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)

            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())

        return wd_params, nowd_params


@SEG_MODELS_REGISTRY.register()
class BiSeNet(nn.Module):
    def __init__(self, n_classes=3, in_channels=3):
        super(BiSeNet, self).__init__()

        self.cp = ContextPath(in_channels=in_channels)
        self.sp = SpatialPath(in_channels=in_channels)
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)

        self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes, up_factor=16)
        self.init_weight()

    def forward(self, x):

        H, W = x.size()[2:]

        feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)

        if self.training:
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)

            return feat_out, feat_out16, feat_out32

        else:
            return feat_out

    def init_weight(self):

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []

        for name, child in self.named_children():

            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params

            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params

        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params










if __name__ == '__main__':
      main(sys.argv)

