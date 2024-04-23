import rospy
import socket
from struct import unpack
from rospy.rostime import Time

rospy.init_node('test')
port = 1212

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("192.168.51.150", port))

jan1_12am_1980 = 315532800 + 5 * 60 * 60 * 24  # 1980 day 1 is Tuesday, but as ADMA time always starts from Sunday, we need to add 5 days to it

while not rospy.is_shutdown():
    data = sock.recvfrom(1000)[0]
    time_ms = unpack('I', data[584:588])[0]
    time_week = unpack('H', data[588:590])[0]
    time_adma = time_ms * 0.001 + time_week * 7 * 24 * 60 * 60 + jan1_12am_1980
    ros_time = rospy.get_rostime()
    diff_ros = ros_time.to_nsec() - time_adma*1e9
    clock_pub = rospy.Publisher('/clock', Time, queue_size=10)
    ros_time = Time(secs=int(time_adma), nsecs=int((time_adma - int(time_adma)) * 1e9))
    clock_pub.publish(ros_time)
    print("Time difference (ROS time - received time):", diff_ros/1e6, "ros_time",ros_time.to_sec())




import rospy
import socket
import subprocess
from struct import unpack


def set_system_time(seconds_since_epoch):
    # Convert the time_adma to a format acceptable by the date command
    time_string = "@{}".format(seconds_since_epoch)
    # Set system time
    subprocess.call(["sudo", "date", "--set", time_string])


rospy.init_node('test')
port = 1212

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#sock.bind(("192.168.12.255", port))
sock.bind(("192.168.51.150", port))

jan1_12am_1980 = 315532800 + 5 * 60 * 60 * 24  # 1980 day 1 is Tuesday, but as ADMA time always starts from Sunday, we need to add 5 days to it

data = sock.recvfrom(1000)[0]
time_ms = unpack('I', data[584:588])[0]
time_week = unpack('H', data[588:590])[0]
time_adma = time_ms * 0.001 + time_week * 7 * 24 * 60 * 60 + jan1_12am_1980
set_system_time(time_adma)
while not rospy.is_shutdown():
    data = sock.recvfrom(1000)[0]
    time_ms = unpack('I', data[584:588])[0]
    time_week = unpack('H', data[588:590])[0]
    time_adma = time_ms * 0.001 + time_week * 7 * 24 * 60 * 60 + jan1_12am_1980
    ros_time = rospy.get_rostime()
    diff_ros = ros_time.to_nsec() - time_adma*1e9
    print("Time difference (ROS time - received time):", diff_ros/1e6, "ros_time",ros_time.to_sec())
    # Reset system time to time_adma


