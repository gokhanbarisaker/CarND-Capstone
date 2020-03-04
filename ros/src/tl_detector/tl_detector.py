#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import torch
import time
import numpy as np

from transforms import decode_results
from dla import get_pose_net
from cls_model import ClassifierNet

class TLDetector(object):
    def __init__(self):
        rospy.loginfo("Initializing TLDetector")
        self.frame_id = 0
        self.history = []
        self.is_red = False

        # ROS declarations
        rospy.init_node('tl_detector', )

        self.camera_image = None
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.bridge = CvBridge()

        # Read config
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.device = self.config['device']
        device = torch.device(self.device)

        # Detector model
        self.detector_max_frequency = self.config['detector_max_frequency']
        self.detection_threshold = self.config['detection_threshold']
        self.last_time = -1
        self.skipped_from_last = 0

        self.K = self.config['max_detections']

        rospy.loginfo("Loading detecor model...")
        self.model = get_pose_net(34, heads={'hm': 1, 'wh': 2}, head_conv=-1).to(self.device)
        state_dict = torch.load("./data/detector.pth", map_location=device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        rospy.loginfo("Loaded detecor model.")

        # Classifier model
        rospy.loginfo("Loading clasification model...")
        self.min_red_light_size = self.config['min_red_light_size']
        self.red_light_threshold = self.config['red_light_threshold']
        self.light_change_history_length = self.config['light_change_history_length']
        self.cmodel = ClassifierNet()
        state_dict = torch.load("./data/classifier.pth", map_location=device)
        self.cmodel.load_state_dict(state_dict)
        self.cmodel  = self.cmodel.to(self.device)
        rospy.loginfo("Loaded clasification model.")

        # Run!
        rospy.spin()

    def image_cb(self, msg):
        """
            Publishes:
                -1 when frame skipped
                0 when no red light detected
                1 red light detected
        """
        RED_LIGHT_VALUE = 1
        NORED_LIGHT_VALUE = 0
        SKIPPED_FRAME_VALUE = -1

        self.has_image = True
        now = time.time()
        if 1/(now-self.last_time)<=self.detector_max_frequency:
            self.skipped_from_last=0
            self.camera_image = msg
            self.last_time = now
            is_red_light, time_ms, detections, clsval = self.is_red_light_visible()
            rospy.loginfo("Time(ms):"+str(int(time_ms)) + 
                " Red?:"+str(is_red_light)+
                " Detections:"+str(detections)+
                " Classificiations:"+str(clsval))
            self.frame_id += 1
            if is_red_light:
                self.upcoming_red_light_pub.publish(Int32(RED_LIGHT_VALUE))
            else:
                self.upcoming_red_light_pub.publish(Int32(NORED_LIGHT_VALUE))
        else:
            self.upcoming_red_light_pub.publish(Int32(SKIPPED_FRAME_VALUE))
            self.skipped_from_last+=1

    def preapre_tensor(self, cv_image):
        cv_image = cv2.resize(cv_image, (512,512))
        resized = cv_image.copy()
        cv_image = (cv_image / 255.0)
        cv_image = cv_image.transpose(2,0,1)

        input = torch.tensor(cv_image, dtype=torch.float).unsqueeze(0)
        return input, resized

    def detect(self, input):
        output = self.model.forward(input.to(self.device))[0]   
        output_hm = output['hm']
        output_wh = output['wh']

        # Decode results
        dets = decode_results(output_hm, output_wh, self.K)[0]
        return dets

    def clip(self, val):
        if val<0:
            return 0
        if val>511:
            return 511
        return val

    def classify(self, input, bbox):
        x1,y1,x2,y2 = bbox
        x1,y1,x2,y2 = self.clip(int(4*x1)),self.clip(int(4*y1)),self.clip(int(4*x2)),self.clip(int(4*y2) )

        if x2-x1<=self.min_red_light_size or y2-y1<=self.min_red_light_size:
            return False, 0.0

        input = input[y1:y2,x1:x2,:].copy()
        input_copy = input.copy()
        input = cv2.resize(input,(32,32))

        input= input - 0.5
        input = (input / 255.0) - 0.5
        input = input.transpose(2,0,1)
        input = torch.tensor(input, dtype=torch.float).unsqueeze(0)

        result = (self.cmodel(input.to(self.device))[0]).cpu().detach().numpy()
        result_val = np.exp(result[1])/(np.exp(result[0])+np.exp(result[1]))
        result = result_val>self.red_light_threshold

        return result, result_val
        
    def is_red_light_visible(self):
        if(not self.has_image):
            return False, 0.0, 0, []

        time_a = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        original_shape = cv_image.shape
        cv_image, cv_resized = self.preapre_tensor(cv_image)
        detections = self.detect(cv_image)

        red_lights_colors = [False]
        scores = detections['scores']
        bboxes = detections['bboxes']
        cls_vals = []
        for di,_ in enumerate(detections):
            if scores[di] > self.detection_threshold:
                classification_result, cls_val = self.classify(cv_resized, bboxes[di])
                red_lights_colors.append(classification_result)
                cls_vals.append(cls_val)
        light_state = any(red_lights_colors)
        self.history.append(light_state)
        self.history = self.history[-self.light_change_history_length:]
        
        if all([h==self.history[-1] for h in self.history]):
            self.is_red = self.history[-1]

        time_b = time.time()
        elapsed_ms = 1000.0 * (time_b-time_a)
        return self.is_red, elapsed_ms, len(detections), cls_vals


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
