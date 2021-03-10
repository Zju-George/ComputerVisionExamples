## 单摄像头三维重建问题：计算平面上某点的3D坐标

### 问题描述

单目摄像头成像后，根据某点 **P** 的像素坐标 **(u, v)**，并已知该点在三维空间中落在一个平面上，例如 **z=0** 平面上，求该点在三维空间的坐标 **(x, y, 0)**。

### 实际案例

例如，拿红外激光笔打光到墙面上一点，想通过图像知道该点的三维坐标。具体来说，拿红外摄像头成像得到灰度图 **IMAGE**。需要寻找 **IMAGE** 上的亮斑，显然在 **IMAGE** 上，亮斑中心以及附近的像素值(亮度)会大于其他像素。结合此性质再利用低通滤波来剔除高频噪声(假光斑点)。若到该亮斑中心像素坐标为 **(u, v)**，又已知该亮斑必然落在墙面上，结合三维重建的知识可推断该亮斑的三维坐标 **(x, y, z=0)** 有且仅有唯一解。

注：检测亮斑的算法可参考[这篇教程](https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/)。

### 项目结构和环境

- 项目结构
  - 实验中的图片均放在 `assets/` 目录下。
  - 代码在 `src/` 目录下。

- 环境配置
  - 语言：**Python>=3.6**
  - 依赖：**opencv-python==4.5.1.48**、**argparse**
  - 可以通过 `python -m pip install -r requirements.txt` 安装。
### 离线步骤

1. 利用张正友相机标定法求出相机内参，包括相机焦距和畸变系数。
   1. 打印下方棋盘格图片，尽量**平铺**于平面上，接着用相机从不同方位拍几张图片。<img src="https://github.com/Zju-George/3DReconstructionExample/raw/main/assets/checkerboard.png" alt="HMI" width="433" height="305" align="bottom" />
   
   2. 将拍的 jpg 图片放置于 `assets/` 下。
   3. 进入 `src/` 目录，执行 `python calibration.py`。
   4. 检验结果的合理性并记录。在 `src/calibration.py` 中，[这行代码](https://github.com/Zju-George/3DReconstructionExample/blob/a2ab1cc6d42094d5043bbdafdee6d1865ed5240b/src/calibration.py#L44)会执行相机标定求解。
        ```python
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        ```
        第一个返回值 `ret` 存储着相机标定的重投影误差(reprojection error)，值越小意味着标定越精确。一般来说`ret` 不应大于 2，而如果值大于 5，上述某步大概率做得有问题。第二个返回值 `mtx` 是相机投影矩阵，第三个返回值 `dist` 是相机的畸变系数。
   

2. 准备PNP(perspective n points)算法需要的数据。PNP算法可用来求解相机外参，相机外参是相机坐标系相对于模型坐标系的变换。特别地，变换可分解为一个平移向量和一个旋转向量。
   1. **固定相机位置**。(**特别注意**：如果相机位置改变，须重新走一遍步骤**2**！)
   2. 在成像场景中放置若干(**最少4个**)可精准定位(包括三维空间的测量与图像空间的像素坐标获取)的标志点。注：因为 PNP 算法的原理是最小二乘优化，所以原则上标志点越多，相机外参计算也会越准确。