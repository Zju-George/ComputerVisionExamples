## 单摄像头三维重建问题：计算平面上某点的3D坐标

### 问题描述

单摄像头成像后，已知某点**P**的像素坐标 **(u, v)**，且已知该点在三维空间中落在一个平面上，例如 **z=0** 平面上，求该点在三维空间的坐标 **(x, y, z=0)**。

### 实际案例

如红外激光笔打到墙面上的一点，用红外摄像头成像得到灰度图 **IMAGE**。寻找 **IMAGE** 的亮斑。显然在 **IMAGE** 上，亮斑中心以及附近的像素值(亮度)会大于其他像素。利用这个性质再结合上高频噪声滤波的方法，检测到该亮斑中心像素坐标为 **(u, v)**。又已知该亮斑必然落在墙面(平面)上，利用三维重建的知识可知，该亮斑的三维坐标 **(x, y, z=0)** 有唯一解。

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

1. 利用张正友相机标定法，计算出相机内参，包括相机焦距和畸变系数。
   1. 打印下图棋盘格，置于平面上，并用相机从不同方位拍几张图片。<img src="https://github.com/Zju-George/3DReconstructionExample/raw/main/assets/checkerboard.png" alt="HMI" width="433" height="305" align="bottom" />
   
   2. 将拍的 jpg 图片放置于 `assets/` 下。
   3. 进入 `src/` 目录，执行 `python calibration.py`。
   4. 检验结果的合理性与记录结果。具体的，`src/calibration.py` 里[这句代码](https://github.com/Zju-George/3DReconstructionExample/blob/a2ab1cc6d42094d5043bbdafdee6d1865ed5240b/src/calibration.py#L44)执行相机标定求解。
        ```python
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        ```
        第一个返回值 `ret` 存储着相机标定的重投影误差(reprojection error)，值越好代表标定越精确。`ret` 一般会小于 2，如果这个值大于 5，很大概率上述哪步出错了。第二个返回值 `mtx` 是相机投影矩阵，第三个返回值 `dist` 是相机的畸变系数。
   

2. 准备PNP(perspective n points)算法需要的数据，PNP算法可以求解相机外参。相机外参是相机坐标系相对于模型坐标系的变换，特别地，可分解为一个平移向量和一个旋转向量。
   1. **固定相机位置**。**(!!如果相机位置后续改变，需要重新执行步骤**2**得到新的相机外参!!)**