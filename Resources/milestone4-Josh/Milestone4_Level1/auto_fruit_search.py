# M4 - Autonomous fruit searching

# basic python packages
import sys, os
from urllib import robotparser
import cv2
import numpy as np2
import json
import ast
import argparse
import time

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
#import pygame # python package for GUI
import shutil # python package for file operations
from util.Helper import *

from operateClass import Operate

# Import path planning algorithm
from util.rrt import RRT


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id - 1][0] = x
                    aruco_true_pos[marker_id - 1][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    
    angle = get_angle_robot_to_goal(robot_pose, np.asarray(waypoint))

    print('\n')
    print('angle from function:')
    print(angle/np.pi*180)  # Turn right is negative, Turn left is positive
    print('\n')

    #angle = angle if (angle > 0) else (angle + 2 * np.pi)
    angle_turn = 0
    turn = 0
    command = []

    if angle > -0.0001 and angle < 0.0001:
        angle_turn = 0
        command = [0, 0]

    elif angle > 0.0001:   # Turn left
        angle_turn = abs(angle)
        command = [0, 1]
        turn = -1

    elif angle < 0.0001:    # Turn right
        angle_turn = abs(angle)
        command = [0, -1]
        turn = 1

    print('\n')
    print('The turning angle is: ')
    print(angle_turn/np.pi*180)
    print('degrees \n')


    wheel_vel = 10 # tick to move the robot

    # print(angle/np.pi*180)
    
    # turn towards the waypoint
    turn_time = angle_turn * baseline / (2 * scale * wheel_vel) + 0.1 # replace with your calculation
    #print(turn_time)
    print("Turning for {:.2f} seconds".format(turn_time))
    lv_rot, rv_rot = ppi.set_velocity(command, turning_tick=wheel_vel, time=turn_time)
    
    # after turning, drive straight to the waypoint
    distance = get_distance_robot_to_goal(robot_pose, np.asarray(waypoint))
    drive_time = distance/(wheel_vel * scale)# replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    lv_forward, rv_forward = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    return lv_rot, rv_rot, lv_forward, rv_forward, turn_time, drive_time   # Return arguments for operate.control (in get_robot_pose), to generate drive_meas


def get_robot_pose(lv_rot, rv_rot, lv_forward, rv_forward, dt_rot, dt_forward):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    
    # Joshua try

    # Rotation
    drive_meas_rot = operate.control(lv_rot, rv_rot, dt_rot)
    operate.update_slam(drive_meas_rot)

    # Forward
    drive_meas_forward = operate.control(lv_forward, rv_forward, dt_forward)
    operate.update_slam(drive_meas_forward)

    # get state
    robot_pose = operate.ekf.get_state_vector() # can try robot.self.state[0], [1] and [2]

    # Joshua end

    # update the robot pose [x,y,theta]
    #robot_pose = [0.0,0.0,0.0] # replace with your calculation
    ####################################################

    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.196')
    parser.add_argument("--port", metavar='', type=int, default=8000)

    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    args, _ = parser.parse_known_args()

    ppi = Alphabot(args.ip,args.port)
    operate = Operate(args, ppi)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    #Set parameters
    goal = np.array([11.5, 9.5])
    start = np.array([0.5, 8.5])

    all_obstacles = [Circle(11.5, 5, 2), Circle(4.5, 2.5, 2), Circle(4.8, 8, 2.5)]

    rrt = RRT(start=start, goal=goal, width=16, height=10, obstacle_list=all_obstacles, expand_dis=20, path_resolution=0.5)

    # Initialise SLAM components and opearte class

    # The following code is only a skeleton code the semi-auto fruit searching task
    while True:
        # enter the waypoints
        # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        

        # robot drives to the waypoint
        waypoint = [x,y]

        lv_rot, rv_rot, lv_forward, rv_forward, turn_time, drive_time = drive_to_point(waypoint,robot_pose)

        # estimate the robot's pose (joshua swapped order with waypoint)
        robot_pose = get_robot_pose(lv_rot, rv_rot, lv_forward, rv_forward, turn_time, drive_time)
        robot_pose = np.transpose(robot_pose)[0]
        deg = robot_pose[2] * 180 / np.pi
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
        print('Angle in degrees: ' + str(deg))
        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break