---

## 🧠 一、基本概念对比

| 类型     | 英文名      | 相机位置                       | 机械臂是否移动相机？ | 应用场景                     |
| -------- | ----------- | ------------------------------ | -------------------- | ---------------------------- |
| 眼在手上 | Eye-in-Hand | 安装在机械臂末端               | ✅ 是                 | 手持相机进行物体识别、抓取等 |
| 眼在手外 | Eye-to-Hand | 固定在外部环境（如工作台上方） | ❌ 否                 | 固定相机引导机器人抓取、定位 |

---

## 🎯 二、标定目标对比

| 类型     | 标定目标                 | 表达式        | 意义                       |
| -------- | ------------------------ | ------------- | -------------------------- |
| 眼在手上 | Camera 到 Gripper 的变换 | $ ^{c}T_{g} $ | 相机相对于机械臂末端的位姿 |
| 眼在手外 | Camera 到 Base 的变换    | $ ^{c}T_{b} $ | 相机相对于机器人基座的位姿 |

---

## 📐 三、数学模型与推导

### 1. Eye-in-Hand（眼在手上）

#### 变换关系：

$$
^{b}T_{m} = ^{b}T_{g} \cdot ^{g}T_{c} \cdot ^{c}T_{m}
$$

其中：
- $ ^{b}T_{g} $：gripper 在 base 下的位姿；
- $ ^{c}T_{m} $：marker 在 camera 下的位姿；
- $ ^{g}T_{c} $：我们要求的 camera 到 gripper 的变换；

整理得：

$$
^{g}T_{c} = (^{b}T_{g})^{-1} \cdot ^{b}T_{m} \cdot (^{c}T_{m})^{-1}
$$

---

### 2. Eye-to-Hand（眼在手外）

#### 变换关系：

$$
^{c}T_{m} = ^{c}T_{b} \cdot ^{b}T_{g} \cdot ^{g}T_{m}
$$

其中：
- $ ^{c}T_{b} $：我们要求的 camera 到 base 的变换；
- $ ^{g}T_{m} $：已知的 gripper 到 marker 的变换（固定）；

整理得：

$$
^{c}T_{b} = ^{c}T_{m} \cdot (^{g}T_{m})^{-1} \cdot (^{b}T_{g})^{-1}
$$

---

## 🛠️ 四、OpenCV 函数调用方式

OpenCV 中使用 `cv2.calibrateHandEye()` 这一个函数来处理两种情况，区别在于输入参数的含义不同。

### ✅ 眼在手上（Eye-in-Hand）

```python
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base=R_base2gripper,
    t_gripper2base=t_base2gripper,
    R_target2cam=R_marker2camera,
    t_target2cam=t_marker2camera,
    method=cv2.CALIB_HAND_EYE_TSAI
)
```

输出的是：
- `R_cam2gripper`: 相机到末端的旋转；
- `t_cam2gripper`: 相机到末端的平移；

---

### ✅ 眼在手外（Eye-to-Hand）

```python
R_cameratobase, t_cameratobase = cv2.calibrateHandEye(
    R_gripper2base=R_base2gripper,
    t_gripper2base=t_base2gripper,
    R_target2cam=R_camera2marker,
    t_target2cam=t_camera2marker,
    method=cv2.CALIB_HAND_EYE_TSAI
)
```

> 注意：`R_target2cam`, `t_target2cam` 应为 marker 到 camera 的变换（即 `R_marker2camera.T`, `-R_marker2camera.T @ t_marker2camera`）

输出的是：
- `R_cameratobase`: 相机到基座的旋转；
- `t_cameratobase`: 相机到基座的平移；

---

## 📌 五、数据采集建议

| 类型        | 数据采集建议                                                 |
| ----------- | ------------------------------------------------------------ |
| Eye-in-Hand | 移动机械臂末端，在不同角度和位置拍摄固定 marker；            |
| Eye-to-Hand | 移动机械臂末端（上面固定 marker），从不同角度出现在相机视野中； |

> ⚠️ 建议采集 10~20 组数据，保证运动多样性，提升标定精度。

---

## 📊 六、完整对比表格

| 特征             | 眼在手上 (Eye-in-Hand)                                       | 眼在手外 (Eye-to-Hand)                                       |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 相机安装位置     | 机械臂末端                                                   | 固定在外部环境                                               |
| 是否随机械臂移动 | ✅ 是                                                         | ❌ 否                                                         |
| 标定目标         | 相机 → 末端工具 (`^cT_g`)                                    | 相机 → 机器人基座 (`^cT_b`)                                  |
| 输入参数含义     | - `R_base2gripper`: 末端在基座下的位姿<br>- `R_marker2camera`: marker 在相机下的位姿 | - `R_base2gripper`: 末端在基座下的位姿<br>- `R_camera2marker`: marker 在相机下的位姿（或其逆） |
| OpenCV 函数      | `cv2.calibrateHandEye(...)`                                  | 同上                                                         |
| 使用算法         | Tsai / Park / Daniilidis 等                                  | 同上                                                         |
| 应用场景         | 抓取、视觉伺服、手持扫描                                     | 固定视觉引导抓取、定位、空间测量                             |

---

## 📘 七、小结

| 类型        | 一句话总结                                       |
| ----------- | ------------------------------------------------ |
| Eye-in-Hand | “我拿着相机去看世界，我要知道相机在我手里哪。”   |
| Eye-to-Hand | “相机看着我，我要知道它在我世界坐标系里的位置。” |

---

