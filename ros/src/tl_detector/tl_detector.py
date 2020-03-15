#!/usr/bin/env python
import rospy

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
from detector import TrafficLightsDetector
from classifier import LightClassifier
    
import cv2
import yaml
import time
import numpy as np
import tensorflow as tf

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

        self.detector_max_frequency = self.config['detector_max_frequency']
        self.detection_threshold = self.config['detection_threshold']
        self.detection_iou = self.config['detection_iou']
        self.light_change_history_length = self.config['light_change_history_length']
        self.min_red_light_size = self.config['min_red_light_size']
        self.red_light_threshold = self.config['red_light_threshold']

        self.last_time = -1
        self.skipped_from_last = 0

        # Detector model
        self.detector = TrafficLightsDetector(score_threshold =  self.detection_threshold, 
                                              iou_threshold = self.detection_iou)

        # Classifier model
        self.classifier = LightClassifier()

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

    def detect(self, input):
        boxed_image, result = self.detector.detect_lights(input)

        dets = {
            'scores': [],
            'bboxes': []
        }

        for bb,score in result:
            dets['scores'].append(score)
            dets['bboxes'].append(bb)
        
        return boxed_image, dets

    def clip(self, val):
        if val<0:
            return 0
        if val>415:
            return 415
        return val

    def classify(self, input, bbox):
        x1,y1,x2,y2 = bbox
        x1,y1,x2,y2 = self.clip(int(x1)),self.clip(int(y1)),self.clip(int(x2)),self.clip(int(y2) )

        if x2-x1<=self.min_red_light_size or y2-y1<=self.min_red_light_size:
            return False, 0.0

        input = input[y1:y2,x1:x2,:].copy()

        if input.shape[0] <= self.min_red_light_size  or input.shape[1] <= self.min_red_light_size:
            return False, 0.0

        input_copy = input.copy()
        input_copy = cv2.resize(input_copy,(32,32))
        input_copy = (input_copy / 255.0) 
        input_copy = np.expand_dims(input_copy, axis=0)


        result = self.classifier.classify(input_copy)[00]
        return result>self.red_light_threshold, result
        
    def is_red_light_visible(self):
        if(not self.has_image):
            return False, 0.0, 0, []

        time_a = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        cv_resized, detections = self.detect(cv_image)

        red_lights_colors = [False]
        scores = detections['scores']
        bboxes = detections['bboxes']
        cls_vals = []
        for di,_ in enumerate(scores):
            if scores[di] > self.detection_threshold:
                classification_result, cls_val = self.classify(cv_resized, bboxes[di])
                red_lights_colors.append(classification_result)
                cls_vals.append(cls_val)
        cv2.imwrite('temp.jpg', cv_resized)
        light_state = any(red_lights_colors)
        self.history.append(light_state)
        self.history = self.history[-self.light_change_history_length:]
        
        if all([h==self.history[-1] for h in self.history]):
            self.is_red = self.history[-1]

        time_b = time.time()
        elapsed_ms = 1000.0 * (time_b-time_a)
        return self.is_red, elapsed_ms, len(scores), cls_vals


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
