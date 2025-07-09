import cv2
import numpy as np
import sys 
from scipy.spatial.transform import Rotation as R

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



#标定板在相机下的位姿，x,y,z,w,x,y,z
CalPose = np.array([
[ -0.122252106667, 0.182124316692, 0.958851754665,-0.337640943263,0.51402346259,-0.340175723879,0.71137820477],
# [0.0286149606109, 0.143701985478,0.781195640564,-0.288267129891, 0.613377300648, -0.433264137275,0.594098086367],
[0.0478008501232,0.0761481076479, 0.500957250595, -0.39606769219,0.497572432956,-0.37971192566,0.671841432689],
[-0.0196635387838, 0.136720150709, 0.903728723526,-0.322747317766,0.558934701078,-0.41419413957,0.641770506919],
[-0.0234980192035, 0.0441578440368, 0.543818295002,-0.467428324163,0.430398962078,-0.500484002192,0.588033382353],
[0.0330494120717, 0.095269843936, 0.534600496292,-0.498409518144,0.652613497856,-0.245995664991,0.51494631511],
[-0.0804712623358, 0.0920680984855, 0.615765333176,-0.236188033533,0.499504587133,-0.526518017496,0.646134008934],
[0.00658114347607, 0.093214802444, 0.516361355782,-0.212510413178,0.494600249712,-0.368998189209,0.757661041387]
], dtype=np.float64)

#机械臂末端在基座下的位姿，x,y,z,w,x,y,z
ToolPose = np.array([
[-0.236,-0.05509,0.53558,-0.330981,0.391670, 0.566735, -0.644871],
# [-0.51298,-0.2772,0.58267,-0.197026,0.272930, 0.651984, -0.679416],
[-0.54876,-0.37756,0.60763,-0.299769,0.391729, 0.637196, -0.592172],
[-0.36611,-0.00692,0.59476, -0.239849,0.338952, 0.636674, -0.649793],
[-0.58781,-0.28772,0.75958,-0.217125,0.430348, 0.733882, -0.478618],
[-0.58912,-0.26807,0.604, 0.418869,-0.157537, -0.712884, 0.539933],
[-0.63407,-0.11454,0.58364, -0.101459,0.408957, 0.626076, -0.656116],
[-0.56332,-0.36295,0.47085,-0.241652,0.420789, 0.492703, -0.722346]
], dtype=np.float64)


# int main() # Translate to Python main execution block
if __name__ == "__main__":

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
		tmp = attitudeVectorToMatrix(ToolPose[i:i+1, :], True, "") # Pass the i-th row as a 1x7 matrix
		'''
		如果是 "眼在手外"（相机固定），请取消下面一行注释
		以将 Gripper 到 Base 的变换取逆，变为 Base 到 Gripper
		'''
		#tmp = np.linalg.inv(tmp)  # Base → Gripper

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
	for i in range(len(vecHc)):
		cheesePos = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64) # 4x1 matrix
		worldPos = vecHb[i] @ Hcb @ vecHc[i] @ cheesePos
		print(f"第{i}组计算出的标定板在世界坐标系中的位姿： {worldPos.T}")

	'''for i in range(len(vecHc)):
		H_base2target = vecHb[i] @ Hcb @ vecHc[i]
		print(f"第 {i} 组计算出的标定板在世界坐标系中的位姿：")
		print(H_base2target)'''

	print("Press Enter to exit...")
	sys.stdin.readline() 

