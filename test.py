import numpy as np
from scipy.spatial.transform import Rotation as R
from UR_Base import UR_BASE

HOST = "192.168.111.10"
robot = UR_BASE(HOST, fisrt_tcp=None)

T_world_camera = np.array([[ 0.66936684, -0.62154663,  0.40697398, -0.65873061],
 [-0.741347,   -0.52304211,  0.42051347, -0.56488244],
 [-0.0485042,  -0.58318671, -0.81088877,  0.88797413],
 [ 0,          0,          0,          1        ]])

# 相机坐标系下的点
point_camera = np.array([0, 0, 0.67, 1])  # 齐次坐标

# 变换到世界坐标
point_world = T_world_camera @ point_camera

# 输出结果
print("Point in world coordinates:", point_world[:3])

current_pose = robot.get_tcp()
print("Current TCP pose:", current_pose)
next_pos = np.hstack((point_world[:3], current_pose[3:]))
print("Next position in robot coordinates:", next_pos)

robot.moveL(next_pos, speed=0.005, acc=0.005)
