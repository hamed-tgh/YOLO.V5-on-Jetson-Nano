#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_coords, check_img_size)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
import cv2
import time

class ObjectDetector:
    def __init__(self, 
                 weights='yolov5n.pt',  # default YOLOv5 small model
                 device='0',            # device to run the model on ('cpu' or 'cuda:0')
                 imgsz=(640, 640),     # inference image size
                 conf_thres=0.25,      # confidence threshold
                 iou_thres=0.45,       # IoU threshold for NMS
                 classes=None,         # filter by class
                 agnostic_nms=False,   # class-agnostic NMS
                 half=True,           # use FP16 half-precision inference
                 dnn=False):           # use OpenCV DNN for ONNX inference
        self.device = select_device(device)
        self.weights = weights
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, fp16=half)
        self.stride = self.model.stride
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.half = half

        # Warm up the model for faster subsequent inferences
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def detect(self, img):
        """
        Detect objects in a given image (as a NumPy array).
        
        Parameters:
        - img: The input image as a NumPy array (BGR format as used in OpenCV).

        Returns:
        - im0: The processed image with detected objects annotated.
        """
        # Pre-process the image
        img0 = img.copy()  # original image
        #img = cv2.resize(img, (640,640))  # resize to model input size
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # Convert to FP16 if half-precision is enabled
        img /= 255  # Normalize to range [0, 1]
        if len(img.shape) == 3:
            img = img[None]  # Add batch dimension

        # Inference
        pred = self.model(img)

        # Apply Non-Max Suppression (NMS)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        # Process predictions
        annotator = Annotator(img0, line_width=3, example=str(self.model.names))

        for det in pred:  # per image
            if len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Annotate the image with detection results
                for *xyxy, conf, cls in det:
                    label = f'{self.model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

        im0 = annotator.result()  # Get annotated image
        return im0

# Usage example:
# Initialize the detector
#detector = ObjectDetector(weights='yolov5s.pt', device='cuda:0')

# Load an image (using OpenCV for example)
#img = cv2.imread('path/to/your/image.jpg')

# Detect objects in the image
#processed_image = detector.detect(img)

# Display the image using OpenCV (optional)
#cv2.imshow("Detected Image", processed_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# In[ ]:


import cv2
import threading
import numpy as np
from jetson_utils import videoOutput


class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False
        self.detector = ObjectDetector(weights='yolov5n.pt', device='cuda:0')
        self.detect = ''

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)


    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.frame = cv2.resize(self.frame, (640,640))
                    self.detect = self.detector(self.frame)
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            detect = self.detect
            grabbed = self.grabbed
        return grabbed, frame , detect 

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )




def run_cameras():
    num = 0 
    font = cv2.FONT_HERSHEY_SIMPLEX
    window_title = "Dual CSI Cameras"
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            capture_width=720,
            capture_height=720,
            flip_method=0,
            display_width=720,
            display_height=720,
            framerate=60
        )
    )
    left_camera.start()
    t0 = time.time()

    if left_camera.video_capture.isOpened() :

        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                _, left_image , processed_image = left_camera.read()
                #print(left_image.shape)
                #left_image = left_image.astype(np.float32)
                
                #print(left_image.shape)
                #processed_image = detector.detect(left_image)
                #print(processed_image.shape)
                
                new_frame_time = time.time()
                FPS =  (1 / (new_frame_time - t0)) * 1
                t0 = new_frame_time
                image_left = cv2.putText(left_image, str(FPS), (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
                
                camera_images = np.hstack((left_image, processed_image))

                #END TEST


                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, camera_images)
                else:
                    break

                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
                elif keyCode == ord("s"):
                    cv2.imwrite("//home//pars//Desktop//Backup//CODE//MINE//Left_Images//"+str(num)+'.png' , left_image)
                    print("CHAP_CHPASHOD")
                    cv2.imwrite("//home//pars//Desktop//Backup//CODE//MINE//Right_Images//"+str(num)+'.png' , right_image)
                    print("RAST_CHPASHOD     " , str(num) )
                    num+=1
                    
        finally:

            left_camera.stop()
            left_camera.release()
            
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to open both cameras")
        left_camera.stop()
        left_camera.release()
        

     
        


# In[ ]:


#


                
if __name__ == "__main__":
    
    run_cameras()  

