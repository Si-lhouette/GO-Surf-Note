#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 请使用python2 运行此代码，因为rosbagd对python3不支持
# 此代码用于以odom为基准，找到与其匹配（时间差不超过topic_time_diff(ms)）的 image & depth image
# 用于从实物的rosbag中生成训练集

import rospy
import cv2
import os
import rosbag
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import pdb
from collections import deque
import tf.transformations as tf
import numpy as np


### Parameters ###
topic_time_diff = 30 # ms
bag_path = '/home/michael/Recon/go_surf_ws/src/go-surf/neural_rgbd_data/FastLab_large/for_recon_2.bag'
output_dir = '/home/michael/Recon/go_surf_ws/src/go-surf/neural_rgbd_data/FastLab_large'
odom_topic_name = '/vins_estimator/camera_pose'
image_topic_name = '/cam_d430/color/image_raw'
depth_topic_name = '/cam_d430/depth/image_rect_raw'
time_range_lb = 11.0
time_range_ub = 19.3
##################

processed_timestamps = set()
err_timestamp_secs = set()
bridge = CvBridge()
bag = rosbag.Bag(bag_path, 'r')

odom_queue = deque(maxlen=3)
gray_image_queue = deque(maxlen=60) # 30 hz
depth_image_queue = deque(maxlen=30) # 15 hz

first_timestamp = None

image_save_path = os.path.join(output_dir, 'images')
depth_save_path = os.path.join(output_dir, 'depth_filtered')
pose_save_path = os.path.join(output_dir, 'trainval_poses.txt')

if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)
if not os.path.exists(depth_save_path):
    os.makedirs(depth_save_path)

write_cnt = 0
for topic, msg, timestamp in bag.read_messages(topics=[odom_topic_name, image_topic_name, depth_topic_name]):
    # 更新一个msg
    timestamp = msg.header.stamp # 这个stamp才是真正准的
    if topic == odom_topic_name:
        odom_queue.append((msg, timestamp))
        timestamp_sec = timestamp.to_sec()
        if first_timestamp is None:
            first_timestamp = timestamp_sec
    elif topic == image_topic_name:
        gray_image_queue.append((msg, timestamp))
        timestamp_sec = timestamp.to_sec()
        if first_timestamp is None:
            first_timestamp = timestamp_sec
    elif topic == depth_topic_name:
        depth_image_queue.append((msg, timestamp))
        timestamp_sec = timestamp.to_sec()
        if first_timestamp is None:
            first_timestamp = timestamp_sec
    
    if not (odom_queue and gray_image_queue and depth_image_queue):
        continue

    # 打印Debug 信息
    print('---------------------------------------------------------')
    print('odom_queue: ')
    for odom, odom_time in odom_queue:
        print('timestamp {:.6f}'.format(odom_time.to_sec() - first_timestamp))

    print('gray_image_queue: ')
    for gray_image, gray_image_time in gray_image_queue:
        print('timestamp {:.6f}'.format(gray_image_time.to_sec() - first_timestamp))

    print('depth_image_queue: ')
    for depth_image, depth_image_time in depth_image_queue:
        print('timestamp {:.6f}'.format(depth_image_time.to_sec() - first_timestamp))


    # use the oldest one of the lowest hz msg to match others
    oldest_odom_msg, oldest_odom_time = odom_queue[0]
    odom_msg = None
    gray_image_msg = None
    gray_image_time_msg = None
    depth_image_msg = None
    depth_image_time_msg = None

    odom_msg = oldest_odom_msg
    odom_time_msg = oldest_odom_time

    ## 尝试匹配gray_image
    min_time_diff = topic_time_diff / 1000.0  # Convert to seconds
    for gray_image, gray_image_time in gray_image_queue:
        time_diff = abs(odom_time_msg.to_sec() - gray_image_time.to_sec())
        if time_diff <= min_time_diff:
            gray_image_msg = gray_image
            gray_image_time_msg = gray_image_time
            min_time_diff = time_diff
    if not gray_image_msg is None:
        print('choose grey timestamp {:.6f}'.format(gray_image_time_msg.to_sec() - first_timestamp))

    ## 尝试匹配depth_image
    min_time_diff = topic_time_diff / 1000.0  # Convert to seconds
    for depth_image, depth_image_time in depth_image_queue:
        time_diff = abs(odom_time_msg.to_sec() - depth_image_time.to_sec())
        if time_diff <= min_time_diff:
            depth_image_msg = depth_image
            depth_image_time_msg = depth_image_time
            min_time_diff = time_diff
    if not depth_image_time_msg is None:
        print('choose depth timestamp {:.6f}'.format(depth_image_time_msg.to_sec() - first_timestamp))


    # 删除过期失效odom
    if odom_msg is None or gray_image_msg is None or depth_image_msg is None:
        if gray_image_queue and depth_image_queue:
            last_gray_image_time = gray_image_queue[-1][1].to_sec()
            last_depth_image_time = depth_image_queue[-1][1].to_sec()
            time_diff1 = last_gray_image_time - odom_time_msg.to_sec()
            time_diff2 = last_depth_image_time - odom_time_msg.to_sec()
            if time_diff1 > 0 and time_diff2 > 0:
                odom_queue.popleft()
                print("\033[91mError, This odom with stamp {:.6f} has no match image\033[0m".format(odom_time_msg.to_sec()))
                err_timestamp_secs.add(odom_time_msg.to_sec())
        continue

    
    timestamp_sec = odom_time_msg.to_sec() - first_timestamp
    if timestamp_sec in processed_timestamps:
        continue


    # 写入训练集
    if timestamp_sec > time_range_lb and timestamp_sec < time_range_ub:
        image_filename = os.path.join(image_save_path, 'image_{:.6f}.png'.format(timestamp_sec))
        image = bridge.imgmsg_to_cv2(gray_image_msg,  desired_encoding='rgb8')
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_filename, image)
        
        depth_filename = os.path.join(depth_save_path, 'depth_{:.6f}.png'.format(timestamp_sec))
        depth = bridge.imgmsg_to_cv2(depth_image_msg,  desired_encoding='16UC1')
        cv2.imwrite(depth_filename, depth)

        with open(pose_save_path, 'a') as f:

            pose = tf.quaternion_matrix(np.array([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]))
        
            pose[0,3] = odom_msg.pose.pose.position.x
            pose[1,3] = odom_msg.pose.pose.position.y
            pose[2,3] = odom_msg.pose.pose.position.z
            np.savetxt(f, pose, fmt='%.6f', delimiter=' ')
        write_cnt += 1

    # 删除成功匹配odom
    processed_timestamps.add(timestamp_sec)
    odom_queue.popleft()

    print('Saved odom timestamp {:.6f}'.format(odom_time_msg.to_sec() - first_timestamp))
    print('Saved grey timestamp {:.6f}'.format(gray_image_time_msg.to_sec() - first_timestamp))
    print('Saved depth timestamp {:.6f}'.format(depth_image_time_msg.to_sec() - first_timestamp))

# Logs
print('---------------------------------------------------------')
print('Err timestamps:')
for err_time in err_timestamp_secs:
    print(err_time)

print("\033[32mWrite {} Msgs.\033[0m".format(write_cnt))
