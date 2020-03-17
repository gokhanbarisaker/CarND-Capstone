#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
import numpy as np
from std_msgs.msg import Int32
from enum import Enum


import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50  # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5


class VehicleState(Enum):
    STOP = 1
    KL = 2


class Behaviour(object):
    @staticmethod
    def get_successor_state(state, closest_idx, stopline_idx, farthest_idx):
        if (stopline_idx == -1) or (stopline_idx >= farthest_idx):
            return VehicleState.KL
        else:
            return VehicleState.STOP


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1
        self.state = VehicleState.KL

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        # The final waypoint is expected to include fixed number of waypoints ahead of the vehicle
        self.final_waypoints_pub = rospy.Publisher(
            'final_waypoints', Lane, queue_size=1)

        self.loop()

    def loop(self):
        # Publishing frequency set to 50 Hz
        # That is enough to feed waypoint follower
        # Waypoint follower from Autoware uses 30 Hz

        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                self.publish_waypoints()

            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        # Query returns [0:position, 1:index]. We care about the index.
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS

        self.state = Behaviour.get_successor_state(
            self.state, closest_idx, self.stopline_wp_idx, farthest_idx)

        final_lane = WaypointUpdater.generate_lane(
            self.state, self.base_waypoints.waypoints, closest_idx, self.stopline_wp_idx, farthest_idx)
        self.final_waypoints_pub.publish(final_lane)

    @staticmethod
    def generate_lane(state, waypoints, closest_idx, stopline_idx, farthest_idx):
        lane = Lane()

        if state is VehicleState.KL:
            lane.waypoints = waypoints[closest_idx:farthest_idx]
        elif state is VehicleState.STOP:
            # We don't really need waypoints further than stopline at this state
            waypoints = waypoints[closest_idx:stopline_idx]
            lane.waypoints = WaypointUpdater.decelerate_waypoints(
                waypoints, closest_idx, stopline_idx)
        else:
            raise NotImplementedError()

        # TODO: ??? why did they removed the header assignment?
        # lane.header = self.base_waypoints.header

        return lane

    @staticmethod
    def decelerate_waypoints(waypoints, closest_idx, stopline_idx):
        """
        Takes slice of waypoints and adjusts speed for each one of them
        """

        temp = []
        if len(waypoints) == 0:
            return temp

        # Aim to stop a little bit earlier than the stopline
        stop_idx = max(stopline_idx - closest_idx - 2, 0)

        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            # Two waypoints back from line so front of the car stops at line
            dist = WaypointUpdater.distance_waypoint(waypoints, i, stop_idx)

            # TODO: This profile might decelerate pretty steep. Consider alternative approaches
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0

            # Set waypoint velocity
            p.twist.twist.linear.x = min(
                vel, WaypointUpdater.get_waypoint_velocity(wp))
            temp.append(p)

        return temp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,
                                  waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        self.base_waypoints = waypoints

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data
        rospy.loginfo("IDx: {}".format(self.stopline_wp_idx))

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        rospy.logwarn("Detected obstacle: {}".format(msg))

    @staticmethod
    def set_waypoint_velocity(wp, speed):
        wp.twist.twist.lienar.x = speed

    @staticmethod
    def get_waypoint_velocity(wp):
        return wp.twist.twist.linear.x

    @staticmethod
    def distance_waypoint(waypoints, wp1, wp2):
        dist = 0

        for i in range(wp1, wp2):
            dist += WaypointUpdater.distance_3d(waypoints[i].pose.pose.position,
                                                waypoints[i+1].pose.pose.position)

        return dist

    @staticmethod
    def distance_3d(a, b):
        return math.sqrt(
            (a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
