# 本程序演示如何使用QCar RealSense摄像头获取RGB和深度数据，实现红色物体的检测、中心定位、深度测距，并将结果可视化。
from pal.products.qcar import QCarRealSense
import time
# import struct
import numpy as np 
import cv2

# --------------------
# 计时参数与方法
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate = 50.0
sampleTime = 1/sampleRate
simulationTime = 10.0
print('采样周期: ', sampleTime)

# --------------------
# 初始化RealSense摄像头，获取RGB和深度数据
myCam1 = QCarRealSense(mode='RGB, Depth')

# HSV过滤红色物体的阈值范围
lower_red = np.array([0,100,100])  # HSV红色下限
upper_red = np.array([10,255,255]) # HSV红色上限

# QLabs缩放参数
QLabsScaling = np.float32(10)   # QLabs坐标与实际坐标的缩放比例
MaxDistance = np.float32(39)    # QLabs中深度摄像头最大读取距离（米）

try:
    while elapsed_time() < simulationTime:
        start = time.time()

        # 读取RGB和深度数据
        myCam1.read_RGB()
        myCam1.read_depth(dataMode='PX')  # 以像素模式读取深度

        # 获取原始数据副本
        rawRGB = myCam1.imageBufferRGB #width x height x 3
        rawDepth = myCam1.imageBufferDepthPX #width x height x 1
        depthCopy = myCam1.imageBufferDepthPX.copy()

        # 1. 显示深度图像的灰度图
        depth_normalized = cv2.normalize(depthCopy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow('Depth Image', depth_normalized)
        # 创建彩色深度图
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        cv2.imshow('Depth Colormap', depth_colormap)
        # 融合深度图和彩色图
        alpha = 0.6
        beta = 1 - alpha
        blended_image = cv2.addWeighted(rawRGB, alpha, depth_colormap, beta, 0)

        # 2. RGB转HSV进行红色区域过滤
        hsv_image = cv2.cvtColor(rawRGB, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
        # print(f"红色区域像素数量: {np.sum(red_mask)}")
        # 检测红色物体
        if np.sum(red_mask) > 0:
            # 计算 mask 的矩
            M = cv2.moments(red_mask)
            # 计算质心坐标，避免m00为0
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                print(f"红色区域中心: cX = {cX}, cY = {cY}")
                # 转换为以图像中心为原点的坐标系
                cX = cX - rawRGB.shape[1] // 2
                cY = cY - rawRGB.shape[0] // 2

            # 3. RGB和深度图像坐标对齐
            # RGB相机FOV为69.4°(水平)×42.5°(垂直)，深度相机FOV为86°×57°
            depth_h_fov = 86
            depth_v_fov = 57
            rgb_h_fov = 69.4
            rgb_v_fov = 42.5
            # 计算校正系数
            h_scale = rgb_h_fov / depth_h_fov
            v_scale = rgb_v_fov / depth_v_fov
            # 坐标校正
            depth_cX = int(cX * h_scale)
            depth_cY = int(cY * v_scale)
            print(f"校正后深度图中心: depth_cX = {depth_cX}, depth_cY = {depth_cY}")
            # 转回原图坐标系
            depth_cX += rawDepth.shape[1] // 2
            depth_cY += rawDepth.shape[0] // 2
            print(f"深度图像源坐标: depth_cX = {depth_cX}, depth_cY = {depth_cY}")

            # 4. 计算物体距离
            # 取深度图像对应坐标的深度值
            centrol_distance = rawDepth[depth_cY, depth_cX][0]
            # 深度值转为实际距离（米）
            real_distance = (centrol_distance * MaxDistance / 255) / QLabsScaling
            # 图像上显示距离信息
            cv2.circle(depth_normalized, (depth_cX, depth_cY), 1, (0, 255, 0), -1)
            cv2.putText(blended_image,
                        f"Distance: {real_distance:.2f}m, {centrol_distance:d}px",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0, 255, 0),2)

        # 显示各类图像结果
        cv2.imshow('Color Mask', red_mask)
        cv2.imshow('Blended Image', blended_image)
        cv2.imshow('Original RGB', rawRGB)
        cv2.imshow('Binary Clean', depth_normalized)

        # 计时与延迟
        end = time.time()
        computationTime = end - start
        sleepTime = sampleTime - (computationTime % sampleTime)
        msSleepTime = int(1000 * sleepTime)
        if msSleepTime <= 0:
            msSleepTime = 1
        cv2.waitKey(msSleepTime)

except KeyboardInterrupt:
    print("用户中断！")

finally:
    myCam1.terminate()
    cv2.destroyAllWindows()