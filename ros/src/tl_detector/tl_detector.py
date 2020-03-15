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

from scipy.spatial import KDTree


class TLDetector(object):
    def __init__(self):
        rospy.loginfo("Initializing TLDetector")
        self.frame_id = 0
        self.history = []
        self.is_red = False

        # ROS declarations
        rospy.init_node('tl_detector', )

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []
        self.last_light_wp = -1

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights',
                                TrafficLightArray, self.traffic_cb)

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
        # List of positions that correspond to the line to stop in front of for a given intersection
        self.stop_line_positions = self.config['stop_line_positions']

        self.last_time = -1
        self.skipped_from_last = 0

        # Detector model
        self.detector = TrafficLightsDetector(score_threshold =  self.detection_threshold, 
                                              iou_threshold = self.detection_iou)

        # Classifier model
        self.classifier = LightClassifier()

        # Run!
        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,
                                  waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """
            Publishes:
                traffic light waypoint index, if found. Otherwise, -1. 
        """

        self.has_image = True

        if not self.detector and not self.classifier:
            return

        now = time.time()

        if 1/(now-self.last_time) <= self.detector_max_frequency:
            self.skipped_from_last = 0
            self.camera_image = msg
            self.last_time = now
            self.frame_id += 1

            light_wp = self.get_traffic_light_stop_wp()

            self.last_light_wp = light_wp

            # TODO: Remove self.frame_id
            # TODO: Prevent publishing when the state is not changed (Apply if the publish creates additional costs)
            # TODO: Calculate the WP and return that
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_light_wp))
            self.skipped_from_last += 1

    def get_traffic_light_stop_wp(self):
        if self.waypoints is None:
            return -1

        closest_light = None
        # traffic light stop line
        line_wp_idx = None

        if self.pose:
            car_position = self.get_closest_waypoint(
                self.pose.pose.position.x, self.pose.pose.position.y)

            # TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Get stop line waypoint index
                line = self.stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])

                # Find closest stop line waypoint index
                d = temp_wp_idx - car_position
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light:
            is_red_light, time_ms, detections, clsval = self.is_red_light_visible()

            # rospy.loginfo("Time(ms):"+str(int(time_ms)) +
            #               " Red?:"+str(is_red_light) +
            #               " Detections:"+str(detections) +
            #               " Classificiations:"+str(clsval))

            return line_wp_idx if is_red_light else -1

        return -1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        # TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

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
        # cv2.imwrite('temp.jpg', cv_resized)
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
