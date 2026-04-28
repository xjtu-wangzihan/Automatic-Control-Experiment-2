# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports
"""
QCar2_Image_Interpretation.py
本程序用于图像理解实验，支持CSI与D435相机的标定（棋盘格法）、图像畸变校正、边缘检测(Canny/Sobel)等功能。
适用于图像理解实验手册相关任务。
"""
from pal.products.qcar import QCarCameras,QCarRealSense, IS_PHYSICAL_QCAR
from hal.utilities.image_processing import ImageProcessing
import time
import numpy as np
import cv2
#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 重定义ImageProcessing类，增加Sobel过滤
class ImageProcessing(ImageProcessing):
    def do_sobel(self, frame):
        """
        对图像应用 Sobel 边缘检测
        
        参数:
            image: 输入图像
            
        返回:
            sobel_combined: 结合了水平和垂直方向的 Sobel 边缘检测结果
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 增加高斯模糊
        blur = cv2.GaussianBlur(gray, self.blur, 0)
        # 计算 X 方向梯度
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        # 计算 Y 方向梯度
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        # 计算梯度幅度
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        # 归一化到 0-255（避免全零帧导致除零）
        max_magnitude = np.max(magnitude)
        if max_magnitude <= 1e-12 or not np.isfinite(max_magnitude):
            sobel_combined = np.zeros_like(gray, dtype=np.uint8)
        else:
            sobel_combined = np.uint8(magnitude * 255.0 / max_magnitude)

        return sobel_combined

#endregion

#region : ImageInterpretation Class Setup

class ImageInterpretation():

    def __init__(self,imageSize,frameRate,streamInfo,gridDims,boxSize):

        # 标定所需图片数量
        self.NUMBER_IMAGES = 15
        # 学生输入参数
        self.imageSize      = imageSize
        self.chessboardDim  = [gridDims[0],gridDims[1]]
        self.frameRate      = frameRate
        self.boxSize        = boxSize
        self.sampleRate     = 1/self.frameRate
        self.calibFinished  = False

        # 相机内参与畸变参数初始化
        self.CSICamIntrinsics = np.eye(3, 3, dtype=np.float32)
        # CSI camera intrinsic matrix at resolution [820, 410] is:
        # [[318.86    0.00  401.34]
        #  [  0.00  312.14  201.50]
        #  [  0.00    0.00    1.00]]
        self.CSIDistParam = np.ones((1,5),dtype= np.float32)
        # CSI camera distorion paramters at resolution [820, 410] are:
        # [[-0.9033  1.5314 -0.0173 0.0080 -1.1659]]

        self.d435CamIntrinsics = np.eye(3,3,dtype= np.float32)
        # D435 RGB camera intrinsic matrix at resolution [640, 480] is:
        # [[455.20    0.00  308.53]
        #  [  0.00  459.43  213.56]
        #  [  0.00    0.00    1.00]]
        self.d435DistParam = np.ones((1,5), dtype= np.float32)
        # D435 RGB camera distorion paramters at resolution [640, 480] are:
        # [[-5.1135e-01  5.4549 -2.2593e-02 -6.2131e-03 -2.0190e+01]]

        # 图像缓存
        self.streamD435 = np.zeros((self.imageSize[1][0],self.imageSize[1][1]))
        self.streamCSI = np.zeros((self.imageSize[0][0],self.imageSize[0][1]))

        # CSI相机接口配置
        enableCameras = [False, False, False, False]
        enableCameras[streamInfo[0]] = True
        self.frontCSI = QCarCameras(
            frameWidth  = self.imageSize[0][0],
            frameHeight = self.imageSize[0][1],
            frameRate   = self.frameRate[0],
            enableRight = enableCameras[0],
            enableBack  = enableCameras[1],
            enableLeft  = enableCameras[2],
            enableFront = enableCameras[3]
        )

        # D435相机接口配置
        self.d435Color = QCarRealSense(
            mode=streamInfo[1],
            frameWidthRGB  = self.imageSize[1][0],
            frameHeightRGB = self.imageSize[1][1],
            frameRateRGB   = self.frameRate[1]
        )

        # 初始化图像处理工具
        self.camCalibTool = ImageProcessing()

        self.SimulationTime = 15

    def camera_calibration(self):

        #region : 相机标定流程
        # 保存采集的标定图片
        savedImages = []
        imageCount = 0
        cameraType = "csi"
        while True:
            startTime = time.time()
            # 先采集CSI，再采集D435
            if cameraType == "csi":
                self.frontCSI.readAll()
                endTime = time.time()
                image = self.frontCSI.csiFront.imageData
                computationTime = endTime-startTime
                sleepTime = self.sampleRate[0] - (computationTime % self.sampleRate[0])
            if cameraType == "D435":
                self.d435Color.read_RGB()
                endTime = time.time()
                image = self.d435Color.imageBufferRGB
                computationTime = endTime-startTime
                sleepTime = self.sampleRate[1] - (computationTime % self.sampleRate[1])
            # 显示当前采集图像
            cv2.imshow("Camera Feed", image)
            msSleepTime = int(1000 * sleepTime)
            if  msSleepTime <= 0:
                msSleepTime = 1
            # 按q保存图片
            if cv2.waitKey(msSleepTime) & 0xFF == ord('q'):
                imageCount += 1
                print("保存图片编号: ", imageCount)
                grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                savedImages.append(grayImage)
                if imageCount == self.NUMBER_IMAGES and cameraType == "csi":
                    print("Implement calibration for CSI camera images: ")

                    # ===== SECTION B1 - CSI Camera Parameter Estimation =====
                    print("Camera calibration for front csi")
                    # Quanser's use of OpenCV for camera calibration
                    
                    self.CSICamIntrinsics, self.CSIDistParam = \
                        self.camCalibTool.calibrate_camera(
                            savedImages,
                            self.chessboardDim,
                            self.boxSize)
                    

                    # Printed output for students
                    text = "CSI camera intrinsic matrix at resolution {} is:"
                    print(text.format(self.imageSize[0][:]))
                    print(self.CSICamIntrinsics)

                    text = ("CSI camera distortion parameters "
                        + "at resolution {} are: ")
                    print(text.format(self.imageSize[0][:]))
                    print(self.CSIDistParam)
                    cameraType = "D435"
                    savedImages = []
                    imageCount = 0
                if imageCount == self.NUMBER_IMAGES and cameraType == "D435":
                    print("Implement calibration for "
                        + "realsense D435 camera images:")

                    # ===== SECTION B2 - D435 Camera Parameter Estimation =====
                    print("Camera calibration for  D435 RGB camera")

                    # Quanser's use of OpenCV for camera calibration
                    self.d435CamIntrinsics, self.d435DistParam = \
                        self.camCalibTool.calibrate_camera(
                            savedImages,
                            self.chessboardDim,
                            self.boxSize
                        )
                    
                    # Printed output for students
                    text = ("D435 RGB camera intrinsic matrix "
                        + "at resolution {} is:")
                    print(text.format(self.imageSize[1][:]) )
                    print(self.d435CamIntrinsics)

                    text = ("D435 RGB camera distortion parameters"
                        + "at resolution {} are: ")
                    print(text.format(self.imageSize[1][:]))
                    print(self.d435DistParam)

                    # Use completed distortion correction on D435 image
                    imageShape = np.shape(image)
                    for count, distImg in enumerate(savedImages):
                        undist = self.camCalibTool.undistort_img(
                            distImg,
                            self.d435CamIntrinsics,
                            self.d435DistParam)
                        cv2.imshow("RectifiedImages", undist)
                        cv2.waitKey(500)
                    break
        print("两个相机均已完成标定！")
        self.calibFinished = True
        cv2.destroyAllWindows()
        #endregion

    def line_detection(self, cameraType):

        #region : 线检测主流程
        currentTime = 0
        t0 = time.time()

        while (currentTime < self.SimulationTime):
            LoopStartTime = time.time()
            currentTime = time.time() - t0
            # 选择相机源
            if cameraType == "csi":
                self.frontCSI.readAll()
                endTime = time.time()
                image = self.frontCSI.csiFront.imageData
                computationTime = endTime - LoopStartTime
                sleepTime = self.sampleRate[0] - (computationTime % self.sampleRate[0])
                cameraIntrinsics = self.CSICamIntrinsics
                cameraDistortion = self.CSIDistParam
            if cameraType == "D435":
                self.d435Color.read_RGB()
                endTime = time.time()
                image = self.d435Color.imageBufferRGB
                computationTime = endTime - LoopStartTime
                sleepTime = self.sampleRate[1] - (computationTime % self.sampleRate[1])
                cameraIntrinsics = self.d435CamIntrinsics
                cameraDistortion = self.d435DistParam

            # ============= 图像校正 =============
            print("对原始图像进行矫正... ")

            # Quanser's use of OpenCV for distortion correction
            cv2.imshow("Raw Image", image)
            undistortedImage = self.camCalibTool.undistort_img(image,cameraIntrinsics,cameraDistortion)
            cv2.imshow("Undistorted Image", undistortedImage)
            # ============= 边缘检测 =============
            print("对校正后图像进行边缘检测...")
            cannyFilteredImage = self.camCalibTool.do_canny(undistortedImage)
            sobelFilteredImage = self.camCalibTool.do_sobel(undistortedImage)
            cv2.imshow("Canny Filter", cannyFilteredImage)
            cv2.imshow("Sobel Filter", sobelFilteredImage)
            # ============= 直线提取 =============
            print("从边缘图像中提取直线...")
            linesImage, _ = self.camCalibTool.extract_lines(cannyFilteredImage,undistortedImage)
            print("显示检测到直线的图像...")
            imageDisplayed = linesImage
            cv2.imshow("Lines Image", imageDisplayed)
            msSleepTime = int(1000 * sleepTime)
            if msSleepTime <= 0:
                msSleepTime = 1
            cv2.waitKey(msSleepTime)
        #endregion

    #region : 关闭相机
    def stop_cameras(self):
        # 停止两个相机的数据流
        self.frontCSI.terminate()
        self.d435Color.terminate()
    #endregion

#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 主程序入口
def main():
    try:
        '''
        INPUTS:
        imageSize           = [[L,W],[L,W]] 2x2 array which specifies the
                                resultion for the CSI camera and the D435
        frameRate           = [CSIframeRate, D435frameRate] 2x1 array for
                                image frame rates for CSI, D435 camera
        streamInfo          = [CSIIndex,D435Stream] 2x1 array to specify the
                                CSI camera index and the D435 image stream
        gridDims            = Specify the number of cells in each direction 
                                of the square chessboard used for calibration
        boxSize             = Float value to specify the size of the cells
                                in the chessboard. Note that the unit of the
                                specified value should be in meter
        '''

        # ======== 摄像头初始化设置 ===========
        cameraInterfacingLab = ImageInterpretation(
            imageSize=[[820,410], [640,480]],
            frameRate=np.array([30, 30]),
            streamInfo=[3, "RGB"],
            gridDims=(6,6),
            boxSize=1
        )

        '''
        可选择实验内容：
        - Calibrate   ：相机标定
        - Line Detect ：线检测
        '''
        camMode = "Line Detect"

    # ========= 相机内参与畸变参数（忽略标定模式直接赋值） =========
        csiCameraMatrix  = np.array([
            [495.84,   0.00, 408.03],
            [0.00, 454.60, 181.21],
            [0.00,   0.00,   1.00]
        ])

        csiDistortionCoefficients = np.array([
            -0.57513,
            0.37175,
            -0.00489,
            -0.00277,
            -0.11136
        ])
        
        d435CameraMatrix = np.array([
            [455.20,   0.00, 308.53],
            [0.00, 459.43, 213.56],
            [0.00,   0.00,   1.00]
        ])

        d435DistortionCoefficients = np.array([
            -5.1135e-02,
            5.4549,
            -2.2593e-02,
            -6.2131e-03,
            -2.0190e+01
        ])

        #region : 不同实验模式
        if camMode == "Calibrate":
            try:
                cameraInterfacingLab.camera_calibration()
                if cameraInterfacingLab.calibFinished == True and camMode == "Calibrate":
                    print("标定完成，关闭相机...")
                    cameraInterfacingLab.stop_cameras()
            except KeyboardInterrupt:
                cameraInterfacingLab.stop_cameras()

        if camMode == "Line Detect":
            try:
                ## 可通过命令行选择相机，这里便于实验直接使用 csi 相机
                text = "请选择用于线检测的相机(csi/D435)："
                cameraType = input(text)
                # cameraType = "csi"
                if cameraType == "csi":
                    cameraInterfacingLab.CSICamIntrinsics = csiCameraMatrix
                    cameraInterfacingLab.CSIDistParam     = csiDistortionCoefficients
                    cameraInterfacingLab.line_detection(cameraType)
                elif cameraType == "D435":
                    cameraInterfacingLab.d435CamIntrinsics = d435CameraMatrix
                    cameraInterfacingLab.d435DistParam     = d435DistortionCoefficients
                    cameraInterfacingLab.line_detection(cameraType)
                else:
                    print("无效的相机类型")
            except KeyboardInterrupt:
                cameraInterfacingLab.stop_cameras()
        #endregion
    finally:
        # if not IS_PHYSICAL_QCAR:
        #     import qlabs_setup
        #     qlabs_setup.terminate()
        input('Experiment complete. Press any key to exit...')

#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 运行主程序
if __name__ == '__main__':
    main()
#endregion