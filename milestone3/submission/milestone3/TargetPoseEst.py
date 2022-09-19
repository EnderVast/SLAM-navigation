# estimate the pose of a target object detected
from pickletools import TAKEN_FROM_ARGUMENT1
from urllib import robotparser
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math
from machinevisiontoolbox import Image

import matplotlib.pyplot as plt
import PIL

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.Resampling.NEAREST)
    target = Image(image)==target_number
    blobs = target.blobs()
    [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = np.array(blobs[0].centroid).reshape(2,)
    box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"
    return box

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most five types of targets in each image
    target_lst_box = [[], [], [], [], []]
    target_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = redapple, 2 = greenapple, 3 = orange, 4 = mango, 5=capsicum, 0 = not_a_target
    img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))
    print(img_vals)
    for target_num in img_vals:
        if target_num > 0:
            try:
                box = get_bounding_box(target_num, base_dir/file_path) # [x,y,width,height]
                pose = image_poses[file_path] # [x, y, theta]
                target_lst_box[target_num-1].append(box) # bouncing box of target
                target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose
            except ZeroDivisionError:
                pass

    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i+1] = {'target': box, 'robot': pose}
        
    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = []
    redapple_dimensions = [0.074, 0.074, 0.087]
    target_dimensions.append(redapple_dimensions)
    greenapple_dimensions = [0.081, 0.081, 0.067]
    target_dimensions.append(greenapple_dimensions)
    orange_dimensions = [0.075, 0.075, 0.072]
    target_dimensions.append(orange_dimensions)
    mango_dimensions = [0.113, 0.067, 0.058] # measurements when laying down
    target_dimensions.append(mango_dimensions)
    capsicum_dimensions = [0.073, 0.073, 0.088]
    target_dimensions.append(capsicum_dimensions)

    target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num-1][2]    # Z 
        
        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose
        # This is the default code which estimates every pose to be (0,0)
        #target_dist = (focal_length * true_height) / box[3]
        #target_X = (box[0]*true_height)/focal_length - robot_pose[0]
        #target_Y = (box[3]*true_height)/focal_length - robot_pose[1]

        focal_length_y = camera_matrix[1][1]
        focal_length_x = camera_matrix[0][0]
        theta = robot_pose[2] #robot_pose[2]
        mu_0 = camera_matrix[0][2]
        alpha = np.arctan((box[0]-320)/focal_length_y)

        target_Y = (focal_length_x * true_height)/box[3] + 0.065
        target_X = target_Y * np.tan(alpha)

        phi = np.arctan(target_X/target_Y)
        A = np.sqrt(target_X ** 2 + target_Y ** 2)

        target_X = A * np.cos(theta - phi) 
        target_Y = A * np.sin(theta - phi)

        #target_X += np.sign(target_X) * 0.08 + robot_pose[0]
        #target_Y += np.sign(target_Y) * 0.08 + robot_pose[1]
        
        target_X += robot_pose[0]
        target_Y += robot_pose[1]

        #target_Y = (focal_length_y*true_height)/box[3] + robot_pose[1]    # Z
        #target_X = ((box[0]-mu_0)*target_Y)/focal_length_x + robot_pose[0] 
        #Z = target_Y

        #target_Y = Z*np.sin(theta)
        #target_X = Z*np.cos(theta)


        #hypotenuse = math.sqrt(target_Y*2 + target_X*2)
        #phi2 = np.arctan(target_X/target_Y)
        #phi = robot_pose[2] - phi2
        #target_Y = hypotenuse*np.sin(phi)
        #target_X = hypotenuse*np.cos(phi)

        target_pose = {'x': target_X, 'y': target_Y}
        
        #target_dist = (focal_length * true_height) / box[1]
        #target_pose = {'x': robot_pose[0] + target_dist * np.cos(robot_pose[2]), 'y': robot_pose[1] + target_dist * np.sin(robot_pose[2])}

        target_pose_dict[target_list[target_num-1]] = target_pose
        ###########################################
    
    return target_pose_dict

# merge the estimations of the targets so that there are at most 1 estimate for each target type
def merge_estimations(target_map):
    target_map = target_map
    redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = [], [], [], [], []
    target_est = {}
    num_per_target = 1 # max number of units per target type. We are only use 1 unit per fruit type
    # combine the estimations from multiple detector outputs
    for f in target_map:    
        for key in target_map[f]:
            if key.startswith('redapple'):
                redapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('greenapple'):
                greenapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('mango'):
                mango_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('capsicum'):
                capsicum_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below is the default solution, which simply takes the first estimation for each target type.
    # Replace it with a better merge solution.
    OFFSET_CORRECTION = 0
    if len(redapple_est) > num_per_target:
        temp_x = []
        temp_y = []
        for n in range(len(redapple_est)):
            temp_x.append(redapple_est[n][0])
            temp_y.append(redapple_est[n][1])
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
        q3, q1 = np.percentile(temp_x, [75 ,25])
        IQR = q3 - q1
        
        upper = np.where(temp_x >= (q3+1.5*IQR))
        temp_x = np.delete(temp_x, upper)
        temp_y = np.delete(temp_y, upper)
        lower = np.where(temp_x <= (q1-1.5*IQR))
        temp_x = np.delete(temp_x, lower)
        temp_y = np.delete(temp_y, lower)
        
        q3y, q1y = np.percentile(temp_y, [75 ,25])
        IQRy = q3y - q1y

        upper = np.where(temp_y >= (q3y+1.5*IQRy))
        temp_x = np.delete(temp_x, upper)
        temp_y = np.delete(temp_y, upper)
        lower = np.where(temp_y <= (q1y-1.5*IQRy))
        temp_x = np.delete(temp_x, lower)
        temp_y = np.delete(temp_y, lower)

        avg_x = np.mean(temp_x)
        avg_y = np.mean(temp_y)
        redapple_est = [np.array([np.array(avg_x), np.array(avg_y)])]
    if len(greenapple_est) > num_per_target:
        temp_x = []
        temp_y = []
        for n in range(len(greenapple_est)):
            temp_x.append(greenapple_est[n][0])
            temp_y.append(greenapple_est[n][1])
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
        q3, q1 = np.percentile(temp_x, [75 ,25])
        IQR = q3 - q1
        
        upper = np.where(temp_x >= (q3+0.5*IQR))
        temp_x = np.delete(temp_x, upper)
        temp_y = np.delete(temp_y, upper)
        lower = np.where(temp_x <= (q1-0.5*IQR))
        temp_x = np.delete(temp_x, lower)
        temp_y = np.delete(temp_y, lower)
        
        q3y, q1y = np.percentile(temp_y, [75 ,25])
        IQRy = q3y - q1y

        upper = np.where(temp_y >= (q3y+1*IQRy))
        temp_x = np.delete(temp_x, upper)
        temp_y = np.delete(temp_y, upper)
        lower = np.where(temp_y <= (q1y-1*IQRy))
        temp_x = np.delete(temp_x, lower) 
        temp_y = np.delete(temp_y, lower)

        avg_x = np.mean(temp_x + OFFSET_CORRECTION)
        avg_y = np.mean(temp_y)
        greenapple_est = [np.array([np.array(avg_x), np.array(avg_y)])]
    if len(orange_est) > num_per_target:
        temp_x = []
        temp_y = []
        for n in range(len(orange_est)):
            temp_x.append(orange_est[n][0])
            temp_y.append(orange_est[n][1])
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
        q3, q1 = np.percentile(temp_x, [75 ,25])
        IQR = q3 - q1
        
        upper = np.where(temp_x >= (q3+1.5*IQR))
        temp_x = np.delete(temp_x, upper)
        temp_y = np.delete(temp_y, upper)
        lower = np.where(temp_x <= (q1-1.5*IQR))
        temp_x = np.delete(temp_x, lower)
        temp_y = np.delete(temp_y, lower)
        
        q3y, q1y = np.percentile(temp_y, [75 ,25])
        IQRy = q3y - q1y

        upper = np.where(temp_y >= (q3y+1.5*IQRy))
        temp_x = np.delete(temp_x, upper)
        temp_y = np.delete(temp_y, upper)
        lower = np.where(temp_y <= (q1y-1.5*IQRy))
        temp_x = np.delete(temp_x, lower)
        temp_y = np.delete(temp_y, lower)

        avg_x = np.mean(temp_x)
        avg_y = np.mean(temp_y)
        orange_est = [np.array([np.array(avg_x), np.array(avg_y)])]
    if len(mango_est) > num_per_target:
        temp_x = []
        temp_y = []
        for n in range(len(mango_est)):
            temp_x.append(mango_est[n][0])
            temp_y.append(mango_est[n][1])
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
        q3, q1 = np.percentile(temp_x, [75 ,25])
        IQR = q3 - q1
        
        upper = np.where(temp_x >= (q3+1.5*IQR))
        temp_x = np.delete(temp_x, upper)
        temp_y = np.delete(temp_y, upper)
        lower = np.where(temp_x <= (q1-1.5*IQR))
        temp_x = np.delete(temp_x, lower)
        temp_y = np.delete(temp_y, lower)
        
        q3y, q1y = np.percentile(temp_y, [75 ,25])
        IQRy = q3y - q1y

        upper = np.where(temp_y >= (q3y+1.5*IQRy))
        temp_x = np.delete(temp_x, upper)
        temp_y = np.delete(temp_y, upper)
        lower = np.where(temp_y <= (q1y-1.5*IQRy))
        temp_x = np.delete(temp_x, lower)
        temp_y = np.delete(temp_y, lower)

        avg_x = np.mean(temp_x)
        avg_y = np.mean(temp_y)
        mango_est = [np.array([np.array(avg_x), np.array(avg_y)])]
    if len(capsicum_est) > num_per_target:
        temp_x = []
        temp_y = []
        for n in range(len(capsicum_est)):
            temp_x.append(capsicum_est[n][0])
            temp_y.append(capsicum_est[n][1])
        temp_x = np.array(temp_x)
        temp_y = np.array(temp_y)
        q3, q1 = np.percentile(temp_x, [75 ,25])
        IQR = q3 - q1
        
        upper = np.where(temp_x >= (q3+1.5*IQR))
        temp_x = np.delete(temp_x, upper)
        temp_y = np.delete(temp_y, upper)
        lower = np.where(temp_x <= (q1-1.5*IQR))
        temp_x = np.delete(temp_x, lower)
        temp_y = np.delete(temp_y, lower)
        
        q3y, q1y = np.percentile(temp_y, [75 ,25])
        IQRy = q3y - q1y

        upper = np.where(temp_y >= (q3y+1.5*IQRy))
        temp_x = np.delete(temp_x, upper)
        temp_y = np.delete(temp_y, upper)
        lower = np.where(temp_y <= (q1y-1.5*IQRy))
        temp_x = np.delete(temp_x, lower)
        temp_y = np.delete(temp_y, lower)

        avg_x = np.mean(temp_x)
        avg_y = np.mean(temp_y)
        capsicum_est = [np.array([np.array(avg_x), np.array(avg_y)])]

    for i in range(num_per_target):
        try:
            target_est['redapple_'+str(i)] = {'x':redapple_est[i][0], 'y':redapple_est[i][1]}
        except:
            pass
        try:
            target_est['greenapple_'+str(i)] = {'x':greenapple_est[i][0], 'y':greenapple_est[i][1]}
        except:
            pass
        try:
            target_est['orange_'+str(i)] = {'x':orange_est[i][0], 'y':orange_est[i][1]}
        except:
            pass
        try:
            target_est['mango_'+str(i)] = {'x':mango_est[i][0], 'y':mango_est[i][1]}
        except:
            pass
        try:
            target_est['capsicum_'+str(i)] = {'x':capsicum_est[i][0], 'y':capsicum_est[i][1]}
        except:
            pass
    ###########################################
        
    return target_est


if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    
    
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    
    # estimate pose of targets in each detector output
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)

    # merge the estimations of the targets so that there are only one estimate for each target type
    target_est = merge_estimations(target_map)
    print(target_est)                 
    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    
    print('Estimations saved!')