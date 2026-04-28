## QCar2_Color_Space.py
# 本示例展示了不同的色彩空间，并使用HSV色彩空间过滤出特定的线条

from pal.utilities.vision import Camera2D
from pal.products.qcar import QCar
from hal.utilities.image_processing import ImageProcessing

import time
import numpy as np
import cv2


# --------------------
# 采样参数
sampleRate = 60
sampleTime = 1/sampleRate
print('采样周期: ', sampleTime)

# --------------------
# 图像参数
imageWidth  = 1640
imageHeight = 820
cameraID    = "2@tcpip://localhost:18963"

# --------------------
# 初始化CSI摄像头
myCam = Camera2D(cameraId=cameraID, frameWidth=imageWidth, frameHeight=imageHeight, frameRate=sampleRate)

# --------------------
# QCar初始化
myCar = QCar(readMode=1, frequency=60)

# 鼠标回调函数：显示像素点的RGB和HSV值
def mouse_callback(event, x, y, flags, param):
	if event == cv2.EVENT_MOUSEMOVE:
		# 获取当前像素的BGR值（OpenCV默认BGR）
		bgr = param[y, x]
		# 转为RGB
		rgb = bgr[::-1]
		# 转为HSV
		hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
		# 控制台打印RGB和HSV值
		print(f"位置: ({x}, {y}) | RGB: {rgb} | HSV: {hsv}")


# --------------------
# 主循环
try:
	while True:
		start = time.time()
		# 采集CSI摄像头的RGB图像
		myCam.read()
		rawRGB = myCam.imageData
		# print("采集到图像，尺寸: ", rawRGB.shape)
		# print(rawRGB[:4, 0:3])
		# cv2.imshow('',rawRGB)
		cv2.imshow('CSI Camera', cv2.resize(rawRGB, (820, 410)))
		cv2.setMouseCallback('CSI Camera', mouse_callback, param=rawRGB)
		# 裁剪图像以提升处理速度
		croppedRGB = rawRGB[524:674, 0:1640]
		cv2.imshow('Cropped RGB', croppedRGB)
		# 设置鼠标回调，显示裁剪区域像素值
		cv2.setMouseCallback('Cropped RGB', mouse_callback, param=croppedRGB)

		# 转为HSV空间并对区域内图像进行阈值分割（HSV空间下）
		hsvBuf = cv2.cvtColor(croppedRGB, cv2.COLOR_BGR2HSV)
		# ===== SECTION - CSI 摄像头数据过滤 =====
		binaryImage = ImageProcessing.binary_thresholding(frame= hsvBuf,
													lowerBounds=np.array([80, 50, 100]),
													upperBounds=np.array([255, 255, 255]))

		cv2.imshow('Binary Image', binaryImage)
		# 将检测到的黄色区域叠加到原始图像上
		binaryImage = binaryImage / 255
		processed = rawRGB.copy()
		processed[524:674, 0:1640, 2] = processed[524:674, 0:1640, 2] + (255 - processed[524:674, 0:1640, 2]) * binaryImage
		processed[524:674, 0:1640, 1] = processed[524:674, 0:1640, 1] * (1 - binaryImage)
		processed[524:674, 0:1640, 0] = processed[524:674, 0:1640, 0] * (1 - binaryImage)

		# 显示检测结果叠加图像
		cv2.imshow('Detection Overlay', cv2.resize(processed, (820, 410)))

		cv2.waitKey(1)
		end = time.time()
		dt = end - start
		print(f"循环周期: {dt:.3f}秒 | 频率: {1/dt:.2f} Hz")

except KeyboardInterrupt:
	print("用户中断！")
		
finally:
	# 释放摄像头和QCar资源
	myCam.terminate()
	myCar.terminate()
# --------------------
