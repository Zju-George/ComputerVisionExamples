import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
img1 = cv2.imread('assets/butterfly.png', 0)  # 原图像
img2 = cv2.imread('assets/scan1.jpg', 0)  # 待搜索图像，下面称目标图像
# 启动SIFT检测器
sift = cv2.SIFT_create()
# 使用SIFT查找关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用FLANN匹配器进行匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)  # 获得匹配结果

# 按照Lowe的比率存储所有好的匹配。
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

# 只有好的匹配点多于10个才查找目标，否则显示匹配不足
if len(good) > MIN_MATCH_COUNT:
    # 获取匹配点在原图像和目标图像中的的位置
    # kp1：原图像的特征点
    # m.queryIdx：匹配点在原图像特征点中的索引
    # .pt：特征点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # 获取变换矩阵，采用RANSAC算法
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    # 图像变换，将原图像变换为检测图像中匹配到的形状
    # 获得原图像尺寸
    h, w = img1.shape
    # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标。
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                     ).reshape(-1, 1, 2)
    # 对角点进行变换
    dst = cv2.perspectiveTransform(pts, M)
    # 画出边框
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 5, cv2.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

# 画出匹配点
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3), plt.title('Result'),
plt.axis('off')
plt.show()