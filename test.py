#!/usr/bin/env python
import rospy
import argparse
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
import math
import numpy as np

def main(args):
    all_map_pub = rospy.Publisher('/sdf', PointCloud2, queue_size=1)
    rospy.init_node('random_map_sensing', anonymous=True)
    # transfer to pcl
    points = []
    for x in range(1,10) :
        points.append([x, 0, 0])
    map = PointCloud2()
    map.height = 1
    map.width = len(points)
    map.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    map.point_step = 12  # 12
    map.row_step = 12 * len(points)
    map.is_bigendian = False
    map.is_dense = False
    map.data = np.asarray(points, np.float32).tostring()
    map.header.frame_id = "world"

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        all_map_pub.publish(map)
        print("pub map")
        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    try:
        main(args)
    except rospy.ROSInterruptException:
        pass