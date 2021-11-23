#!/usr/bin/env python
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist

def callback(data):
  # get parameters
  fov_size = rospy.get_param("~fov_size")
  linear_speed = rospy.get_param("~linear_speed")
  rotate_speed = rospy.get_param("~rotate_speed")
  stop_dist = rospy.get_param("~stop_distance")
  accum_list = []
  print "view size {}, linear speed: {}, rotate_speed: {}, stop distance: {}".format(fov_size, linear_speed, rotate_speed, stop_dist)
  # check point cloud data, calculate the average distance in the front of bot.
  # use fov_size to constrain the focused area in the front
  for p in point_cloud2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
    if abs(p[0]) < fov_size and abs(p[1]) < fov_size:
      accum_list.append(p[2])
  if accum_list:
    avg_dist = sum(accum_list) / len(accum_list)
  else:
    avg_dist = 2 * stop_dist
  #pub = rospy.Publisher("/cmd_vel_mux/input/navi", Twist, queue_size=10)
  pub = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=10)
  msg = Twist()
  msg.linear.x = linear_speed if avg_dist > stop_dist else 0
  msg.angular.z = rotate_speed if avg_dist < stop_dist else 0
  print "linear: {}, angular: {}".format(msg.linear.x, msg.angular.z)
  pub.publish(msg)
  

def wander():
  rospy.init_node("my_wander_bot")
  rospy.Subscriber("/camera/depth/points", PointCloud2, callback)
  rospy.spin()

if __name__ == "__main__":
  wander()
    
