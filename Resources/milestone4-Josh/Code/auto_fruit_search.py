# M4 - Autonomous fruit searching

# basic python packages
from curses import ALL_MOUSE_EVENTS
from ipaddress import AddressValueError
from lib2to3.pgen2.tokenize import generate_tokens
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

from operateClass import *

# Import path planning algorithm
from util.rrt import RRT
from util.rrtc import RRTC
from util.a_star import *
from util.d_star import *


actual_turn_time_bias = 0.05
actual_drive_time_bias = 0
rr_scale = 1
resolution = 0.5


EKF_drive_time_bias = 0
EKF_turn_time_bias = 0

fileB = "calibration/param/baseline.txt"
baseline = np.loadtxt(fileB, delimiter=',')

robot_radius = (baseline/2*10) * rr_scale

robot_pose = [0.0,0.0,0.0]


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

    wheel_vel = 15 # tick to move the robot

    # print(angle/np.pi*180)
    operate.take_pic()
    time.sleep(0.2)
    # turn towards the waypoint
    turn_time = angle_turn * baseline / (2 * scale * wheel_vel) # replace with your calculation
    #print(turn_time)
    print("Turning for {:.2f} seconds".format(turn_time))
    lv_rot, rv_rot = ppi.set_velocity(command, turning_tick=wheel_vel, time=turn_time + actual_turn_time_bias)


    # Rotation EKF
    drive_meas_rot = operate.control(lv_rot, rv_rot, turn_time + EKF_drive_time_bias)
    operate.update_slam(drive_meas_rot)
    updateDisplay(operate)

    
    operate.take_pic()
    time.sleep(0.2)
    # after turning, drive straight to the waypoint
    distance = get_distance_robot_to_goal(robot_pose, np.asarray(waypoint))
    print(distance)
    drive_time = distance/(wheel_vel * scale)# replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    lv_forward, rv_forward = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time + actual_drive_time_bias)


    # Forward EKF
    drive_meas_forward = operate.control(lv_forward, rv_forward, drive_time + EKF_drive_time_bias)
    operate.update_slam(drive_meas_forward)
    updateDisplay(operate)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
       # Return arguments for operate.control (in get_robot_pose), to generate drive_meas


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    
    # get state
    robot_pose = operate.ekf.get_state_vector() # can try robot.self.state[0], [1] and [2]
    updateDisplay(operate)

    robot_pose = robot_pose[0:3]
    robot_pose = np.transpose(robot_pose)[0]
    deg = robot_pose[2] * 180 / np.pi

    angle_ekf = robot_pose[2]
    if (robot_pose[2] > 2*math.pi):
        angle_ekf = robot_pose[2] - 2*math.pi
    elif (robot_pose[2] < -2*math.pi):
        angle_ekf = robot_pose[2] + 2*math.pi

    operate.ekf.robot.state[2] = angle_ekf

    print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
    print('Angle in degrees: ' + str(deg))

    ####################################################

    return robot_pose

def spin(n, m):
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')

    wheel_vel = 15 # tick to move the robot
    for j in range(m):
        for i in range(n):
            # print(angle/np.pi*180)
            command = [0, -1]
            angle_turn = 2*math.pi/n
            operate.take_pic()
            time.sleep(1)
            # turn towards the waypoint
            turn_time = angle_turn * baseline / (2 * scale * wheel_vel) # replace with your calculation
            #print(turn_time)
            print("Turning for {:.2f} seconds".format(turn_time))
            lv_rot, rv_rot = ppi.set_velocity(command, turning_tick=wheel_vel, time=turn_time + actual_turn_time_bias)
            ppi.set_velocity([0, 0])
            # Rotation EKF
            drive_meas_rot = operate.control(lv_rot, rv_rot, turn_time + EKF_turn_time_bias)
            operate.update_slam(drive_meas_rot)
            updateDisplay(operate)

            robot_pose = get_robot_pose()

            time.sleep(0.5)

def appendObstacleFruit(arucoPosition, fruitPosition):
    ox = []
    oy = []

    # Generate bounds 
    for i in range(-16, 16):
        ox.append(i)
        oy.append(16)
    for i in range(-16, 16):
        ox.append(-16)
        oy.append(i)
    for i in range(-16, 16):
        ox.append(i)
        oy.append(-16)
    for i in range(-16, 16):
        ox.append(16)
        oy.append(i)

    # Append obstacles
    for i in range(len(arucoPosition)):
        ox.append(arucoPosition[i][0]*10)
        oy.append(arucoPosition[i][1]*10)

    for i in range(len(fruitPosition)):
        ox.append(fruitPosition[i][0]*10)
        oy.append(fruitPosition[i][1]*10)

    for a in range(len(aruco_true_pos)):
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                x = aruco_true_pos[a][0]*10 + j
                y = aruco_true_pos[a][1]*10 + i
                ox.append(x)
                oy.append(y)

    return ox, oy

def generate_waypoints(a_star, sx, sy, gx, gy):
    rx,ry = a_star.planning(sx,sy,gx,gy)

    waypoints = []
    # Get waypoints
    for i in range(len(rx)):
        waypoints.append([rx[i]/10, ry[i]/10])
    waypoints = waypoints[::-1]
    return waypoints

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.70.150')
    parser.add_argument("--port", metavar='', type=int, default=8000)

    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    args, _ = parser.parse_known_args()

    ppi = Alphabot(args.ip,args.port)
    operate = Operate(args, ppi)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    operate.initAruco(aruco_true_pos)

    waypoint = [0.0,0.0]
    

    ox, oy = appendObstacleFruit(aruco_true_pos, fruits_true_pos)
    

    spin(12, 1)
    robot_pose = get_robot_pose()
    print(robot_pose)
        
    a_star = AStarPlanner(ox=ox, oy=oy, resolution=resolution, rr=robot_radius)

    while 1:
        for i in range(len(fruits_true_pos)):   # Loop through each fruit
            start = np.array(robot_pose) * 10
            goal = np.array(fruits_true_pos[i]) * 10 -1

            # Simulate 
            #simulate_astar(ox, oy, start[0], start[1], goal[0], goal[1], robot_radius, resolution)

            # Testing:
            #robot_pose = waypoints[0]

            # Go through each waypoint
            goal_reached = 0
            counter = 0

            waypoints = generate_waypoints(a_star, start[0], start[1], goal[0], goal[1])
            while (goal_reached == 0):
                
                for i in range(1, len(waypoints)):
                    if (i >= 8):    # Generate new waypoints and 
                        start = np.array(robot_pose) * 10
                        waypoints = generate_waypoints(a_star, start[0], start[1], goal[0], goal[1])
                        #simulate_astar(ox, oy, start[0], start[1], goal[0], goal[1], robot_radius, resolution)
                        spin(12, 1)
                        robot_pose = get_robot_pose()
                        break

                    else:   # Run robot
                        waypoint = waypoints[i]
                        drive_to_point(waypoint,robot_pose)
                        robot_pose = get_robot_pose()
                        ppi.set_velocity([0, 0])
                        time.sleep(0.4)
                        
                        if (i == len(waypoints) - 1):
                            goal_reached = 1
                            print("\n")
                            print("Fruit reached")
                            print("\n")
                            time.sleep(3)
                            spin(12,1)
                            robot_pose = get_robot_pose()
                        

            # for j in range(len(waypoints)-2,-1,-1): #for j in range(len(waypoints)-1,-1,-1):
            #     counter = counter + 1

            #     waypoint = waypoints[j]
            #     drive_to_point(waypoint,robot_pose)

            #     robot_pose = get_robot_pose()
                
            #     # exit
            #     ppi.set_velocity([0, 0])
            #     time.sleep(0.4)

            #     if (counter == 4):
            #         counter = 0
            #         spin(12, 1)



            # print("Fruit reached")
            # time.sleep(3)
            # spin(12,1)
            
        break
        #Set parameters

    print("Done!")
        
