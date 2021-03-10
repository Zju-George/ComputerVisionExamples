## 单目摄像头三维重建问题：计算平面上某点的3D坐标

### 问题描述

单目摄像头成像，根据平面上(例如 **z=0** 平面)某点 **P** 的像素坐标 **(u, v)**，求该点在三维空间的坐标 **(x, y, 0)**。

### 实际案例

例如，拿红外激光笔射一束光到墙面上，形成某光点，想通过图像知道该点的三维坐标。

具体来说，记红外摄像头成像得到的灰度图为 **IMAGE**，第一步是寻找 **IMAGE** 上的亮斑。显然在 **IMAGE** 上，亮斑中心以及附近的像素值(亮度)会大于其他像素。结合此性质，再利用低通滤波来剔除高频噪声(假光斑点)，可得到该亮斑中心的像素坐标为 **(u, v)**，又已知该亮斑在墙面上，结合三维重建的知识可推断该亮斑的三维坐标 **(x, y, z=0)** 有且仅有唯一解。

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
        第一个返回值 `ret` 存储着相机标定的重投影误差(reprojection error)，值越小意味着标定越精确。一般来说`ret` 不应大于 2，但如果值大于 5，大概率是上述某步做得有问题。第二个返回值 `mtx` 是相机投影矩阵，第三个返回值 `dist` 是相机的畸变系数。
   

2. 准备PNP(perspective n points)算法所需要的数据。PNP算法用来求解相机外参，相机外参是相机坐标系相对于模型坐标系的变换。特别地，变换可分解为一个平移向量和一个旋转向量，这就是我们要求的相机外参。
   1. **固定相机位置**。(**特别注意**：如果相机位置改变，须重新走一遍步骤**2**！)
   2. 在三维场景中放置若干(**最少4个**)方便精准定位(包括实地测量与像素坐标获取)的标识点。注： PNP 算法“背后”是最小二乘优化。因此，原则上标识点越多，相机外参计算也会越准确。
   3. 尽可能准确地测量标识点的三维坐标。例如下图，将三维坐标系原点设置为窗户左下角，并建立右手坐标系。测量并记录 1-6 点的三维坐标。
        <img src="https://github.com/Zju-George/3DReconstructionExample/raw/main/assets/image.jpg" alt="HMI" width="640" height="480" align="bottom" />