import cv2
import numpy as np
import sys 
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import cv2
# 提示没有aruco的看问题汇总
import cv2.aruco as aruco
from UR_Base import UR_BASE
import json

# 配置摄像头与开启pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
# 获取对齐的rgb和深度图
def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    # 获取intelrealsense参数
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    # 内参矩阵，转ndarray方便后续opencv直接使用
    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
    ])
    # 深度图-16位
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # 深度图-8位
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    pos = np.where(depth_image_8bit == 0)
    depth_image_8bit[pos] = 255
    # rgb图
    color_image = np.asanyarray(color_frame.get_data())
    # return: rgb图，深度图，相机内参，相机畸变系数(intr.coeffs)
    return color_image, depth_image, intr_matrix, np.array(intr.coeffs)


def ur_rxryrz_to_euler(rx, ry, rz):
    rotvec = np.array([rx, ry, rz])
    rot = R.from_rotvec(rotvec)
    euler_zyx = rot.as_euler('zyx', degrees=True)  # 返回 yaw(Z), pitch(Y), roll(X)
    return euler_zyx

def euler_zyx_to_quaternion(yaw, pitch, roll):
    # 输入单位是度，转换为弧度
    r = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    q = r.as_quat()  # 返回 [x, y, z, w]
    # scipy的四元数顺序是 [x, y, z, w]，我们转成 w, x, y, z 顺序返回
    w = q[3]
    x = q[0]
    y = q[1]
    z = q[2]
    return w, x, y, z

def quaternion_to_euler(w, x, y, z):
    """
    四元数转欧拉角（单位：度）
    欧拉角顺序为 ZYX：yaw (Z), pitch (Y), roll (X)
    """
    # 计算欧拉角（单位：弧度）
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # 弧度转度
    roll_x = np.degrees(roll_x)
    pitch_y = np.degrees(pitch_y)
    yaw_z = np.degrees(yaw_z)

    return yaw_z, pitch_y, roll_x  # 顺序为 ZYX



#R和T转RT矩阵
def R_T2RT(R, T):
	# Mat RT; # Python doesn't require explicit declaration like this
	# Mat_<double> R1 = (cv::Mat_<double>(4, 3) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
	# 	R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
	# 	R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
	# 	0.0, 0.0, 0.0);
	R1 = np.vstack((R, np.array([[0.0, 0.0, 0.0]], dtype=np.float64)))

	# cv::Mat_<double> T1 = (cv::Mat_<double>(4, 1) << T.at<double>(0, 0), T.at<double>(1, 0), T.at<double>(2, 0), 1.0);
	T1 = np.vstack((T, np.array([[1.0]], dtype=np.float64)))

	# cv::hconcat(R1, T1, RT);//C=A+B左右拼接
	RT = cv2.hconcat([R1, T1]) # Use cv2.hconcat or np.hstack
	return RT

#RT转R和T矩阵
def RT2R_T(RT): 
	R = RT[0:3, 0:3] 
	T = RT[0:3, 3:4] 
	return R, T


#判断是否为旋转矩阵
def isRotationMatrix(H):
    R = H[0:3, 0:3]          # 取左上角3x3矩阵
    Rt = np.transpose(R)
    shouldBeIdentity = Rt @ R
    I = np.eye(3)
    diff = np.linalg.norm(I - shouldBeIdentity)
    return diff < 1e-6



def eulerAngleToRotatedMatrix(eulerAngle, seq):
	assert eulerAngle.shape == (1, 3), "Input eulerAngle must be a 1x3 matrix/array"
	eulerAngle_rad = eulerAngle / 180.0 * np.pi 
	m = eulerAngle_rad[0]
	rx, ry, rz = m[0], m[1], m[2]
	xs, xc = np.sin(rx), np.cos(rx)
	ys, yc = np.sin(ry), np.cos(ry)
	zs, zc = np.sin(rz), np.cos(rz)
	rotX = np.array([[1, 0, 0], [0, xc, -xs], [0, xs, xc]], dtype=np.float64)
	rotY = np.array([[yc, 0, ys], [0, 1, 0], [-ys, 0, yc]], dtype=np.float64)
	rotZ = np.array([[zc, -zs, 0], [zs, zc, 0], [0, 0, 1]], dtype=np.float64)
	rotMat = None 
	if seq == "zyx":
		rotMat = rotX @ rotY @ rotZ # Use @ for matrix multiplication
	elif seq == "yzx":
		rotMat = rotX @ rotZ @ rotY
	elif seq == "zxy":
		rotMat = rotY @ rotX @ rotZ
	elif seq == "xzy":
		rotMat = rotY @ rotZ @ rotX
	elif seq == "yxz":
		rotMat = rotZ @ rotX @ rotY
	elif seq == "xyz":
		rotMat = rotZ @ rotY @ rotX
	else:
		# Translate cv::error to Python exception
		raise ValueError("Euler angle sequence string is wrong.")
	if not isRotationMatrix(rotMat):
		# Translate cv::error to Python exception
		raise ValueError("Euler angle can not convert to rotated matrix")
	return rotMat

def quaternionToRotatedMatrix(q):
	w, x, y, z = q[0], q[1], q[2], q[3]
	x2, y2, z2 = x*x, y*y, z*z
	xy, xz, yz = x*y, x*z, y*z
	wx, wy, wz = w*x, w*y, w*z
	res = np.array([
		[1 - 2 * (y2 + z2),	2 * (xy - wz),		2 * (xz + wy)],
		[2 * (xy + wz),		1 - 2 * (x2 + z2),	2 * (yz - wx)],
		[2 * (xz - wy),		2 * (yz + wx),		1 - 2 * (x2 + y2)],
	], dtype=np.float64)
	return res

def attitudeVectorToMatrix(m, useQuaternion, seq):
	assert m.size == 6 or m.size == 7, "Input matrix m must have 6 or 7 elements"
	if m.shape[1] == 1:
		m = m.T # Transpose to a row vector
	tmp = np.eye(4, 4, dtype=np.float64)
	if useQuaternion: # normalized vector, its norm should be 1.
		quaternionVec = m[0, 3:7] # Get the quaternion {w, x, y, z}
		tmp[0:3, 0:3] = quaternionToRotatedMatrix(quaternionVec) # Assign the 3x3 rotation matrix
	else:
		rotVec = None # Initialize
		if m.size == 6:
			rotVec = m[0, 3:6] # Get the rotation vector/Euler angles
		elif m.size == 7:
			rotVec = m[0, 7:10] # Translate the original slicing exactly
		if seq == "": # Check if seq is empty string
			rotation_matrix, _ = cv2.Rodrigues(rotVec) # Rodrigues returns R and Jacobian
			tmp[0:3, 0:3] = rotation_matrix # Assign the 3x3 rotation matrix
		else:
			tmp[0:3, 0:3] = eulerAngleToRotatedMatrix(rotVec.reshape(1, -1), seq) # Reshape rotVec to 1x3 for the function

	tmp[0:3, 3:4] = m[0:1, 0:3].T # Assign the transposed translation vector

	# //std::swap(m,tmp); # Comment translated
	return tmp



def get_matrix(CalPose, ToolPose, select_mode, mode):
	R_gripper2base = [] # Use Python lists
	t_gripper2base = []
	R_target2cam = []
	t_target2cam = []
	R_cam2gripper = np.zeros((3, 3), dtype=np.float64) # Initialize with zeros
	t_cam2gripper = np.zeros((3, 1), dtype=np.float64) # Initialize with zeros
	num_images = len(ToolPose) # Get the number of images from ToolPose
	vecHb = []
	vecHc = []
	Hcb = None 
	tempR = None
	tempT = None
	for i in range(num_images): # //计算标定板位姿
		tmp = attitudeVectorToMatrix(CalPose[i:i+1, :], True, "") 
		vecHc.append(tmp)
		tempR, tempT = RT2R_T(tmp) 
		R_target2cam.append(tempR)
		t_target2cam.append(tempT)
	for i in range(num_images): # //计算机械臂位姿
		# 眼在手上
		matrix = attitudeVectorToMatrix(ToolPose[i:i+1, :], True, "") # Pass the i-th row as a 1x7 matrix
		'''
		如果是 "眼在手外"(相机固定)
		以将 Gripper 到 Base 的变换取逆，变为 Base 到 Gripper
		'''
		if select_mode == mode[0]:
			tmp = matrix
		elif select_mode == mode[1]:
			tmp = np.linalg.inv(matrix)  # Base → Gripper

		vecHb.append(tmp)
		tempR, tempT = RT2R_T(tmp) 
		R_gripper2base.append(tempR)
		t_gripper2base.append(tempT)
	R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
		R_gripper2base, t_gripper2base,
		R_target2cam, t_target2cam,
		cv2.CALIB_HAND_EYE_TSAI 
	)
	
	Hcb = R_T2RT(R_cam2gripper, t_cam2gripper) # //矩阵合并
	print("Hcb 矩阵为： ")
	print(Hcb)
	print("是否为旋转矩阵：", isRotationMatrix(Hcb), "\n") # //判断是否为旋转矩阵
	print("第一组和第二组手眼数据验证：") # //Tool_In_Base*Hcg*Cal_In_Cam，用第一组和第二组进行对比验证
	print(vecHb[0] @ Hcb @ vecHc[0])
	print(vecHb[1] @ Hcb @ vecHc[1], "\n") # //.inv()
	print("标定板在相机中的位姿：") # //标定板在相机中的位姿：
	print(vecHc[1])
	print("手眼系统反演的位姿为：") # //手眼系统反演的位姿为：
	print(np.linalg.inv(Hcb) @ np.linalg.inv(vecHb[1]) @ vecHb[0] @ Hcb @ vecHc[0], "\n") # //用手眼系统预测第一组数据中标定板相对相机的位姿，是否与vecHc[1]相同
	print("----手眼系统测试----") # //----手眼系统测试----
	print("机械臂下标定板XYZ为：") # //机械臂下标定板XYZ为：

	if select_mode == mode[0]:
		for i in range(len(vecHc)):
			worldPos = vecHb[i] @ Hcb @ vecHc[i]
			print(f"第{i}组计算出的标定板在世界坐标系中的位姿：")
			print(worldPos)

	elif select_mode == mode[1]:
		for i in range(len(vecHc)):
			H_base2target = vecHb[i] @ Hcb @ vecHc[i]
			print(f"第 {i} 组计算出的标定板在世界坐标系中的位姿：")
			print(H_base2target)
	
	return Hcb


if __name__ == "__main__":

	mode = ["Eye in Hand","Eye to Hand"]
	select_mode = mode[1]

	# 连接机械臂
	CalPose = []
	ToolPose = []
	HOST = "192.168.111.10"
	robot = UR_BASE(HOST, fisrt_tcp=None)
	while True:
		rgb, depth, intr_matrix, intr_coeffs = get_aligned_images()
		# 获取dictionary, 这里使用的是原始的aruco字典
		# 如果需要使用其他字典，可以修改aruco.DICT_ARUCO_OR
		aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
		# 创建detector parameters
		parameters = aruco.DetectorParameters_create()
		# 输入rgb图, aruco的dictionary, 相机内参, 相机的畸变参数
		corners, ids, rejected_img_points = aruco.detectMarkers(rgb, aruco_dict, parameters=parameters,cameraMatrix=intr_matrix, distCoeff=intr_coeffs)

		# aruco码的边长，单位是米
		rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.094, intr_matrix, intr_coeffs)

		key = cv2.waitKey(1)
		# 按键盘q退出程序
		if key & 0xFF == ord('q') or key == 27:
			pipeline.stop()
			break

		if rvec is not None and tvec is not None:
			print("Ids:\n", ids)
			print("Corners:\n", corners)
			print("Rvec:\n", rvec)
			print("Tvec:\n", tvec)
			print(len(ToolPose), len(CalPose))
			# 在图片上标出aruco码的位置
			aruco.drawDetectedMarkers(rgb, corners)
			# 根据aruco码的位姿标注出对应的xyz轴, 0.05对应length参数，代表xyz轴画出来的长度 
			try:
				aruco.drawAxis(rgb, intr_matrix, intr_coeffs, rvec, tvec, 0.05)
			except Exception as e:
				print("Error drawing axis:", e)

			cv2.imshow('RGB image', rgb)
			# 按键盘s保存图片
			if key == ord('s'):
				tcp = robot.get_tcp()
				cal = np.hstack((tvec[0][0], rvec[0][0])).tolist()

				ToolPose.append(tcp)
				CalPose.append(cal)
				print("添加成功")

		else:
			cv2.imshow('RGB image', rgb)

	cv2.destroyAllWindows()

	print("标定板在相机下的位姿：")
	print(CalPose)
	print("机械臂末端在基座下的位姿：")
	print(ToolPose)

	if select_mode == mode[0]:
		with open('in_Poses.json', 'w') as f:
			json.dump({
				'CalPose': CalPose,
				'ToolPose': ToolPose
			}, f)

		with open('in_Poses.json','r') as f:
			data = json.load(f)

	elif select_mode == mode[1]:
		with open('to_Poses.json', 'w') as f:
			json.dump({
				'CalPose': CalPose,
				'ToolPose': ToolPose
			}, f)

		with open('to_Poses.json','r') as f:
			data = json.load(f)
	else:
		print("请正确选择模式!")
		sys.exit(0)
	
	CalPose = data['CalPose']
	ToolPose = data['ToolPose']

	for i in range(len(CalPose)):
		cal = CalPose[i]
		tol = ToolPose[i]

		rx,ry,rz = tol[3], tol[4], tol[5]
		yaw, pitch, roll = ur_rxryrz_to_euler(rx, ry, rz)
		w,x,y,z = euler_zyx_to_quaternion(yaw, pitch, roll)
		ToolPose[i] = [tol[0], tol[1], tol[2], w, x, y, z]

		rvec = np.array(cal[3:6])
		rmat, _ = cv2.Rodrigues(rvec)
		x,y,z,w = R.from_matrix(rmat).as_quat()
		CalPose[i] = [cal[0], cal[1], cal[2], w, x, y, z]


	CalPose = np.array(CalPose, dtype=np.float64)
	ToolPose = np.array(ToolPose, dtype=np.float64)

	print(CalPose)
	print(ToolPose)

	# 计算手眼标定矩阵
	matrix = get_matrix(CalPose, ToolPose, select_mode, mode)
	print(matrix)


