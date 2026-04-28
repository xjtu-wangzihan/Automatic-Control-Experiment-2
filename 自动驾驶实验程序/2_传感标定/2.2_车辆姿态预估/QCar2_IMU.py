# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#region : File Description and Imports
"""
QCar2_IMU.py
本程序用于QCar2小车IMU实验，演示如何读取IMU（加速度计、陀螺仪）数据，分析噪声特性，并通过积分估算小车位置。
支持实时可视化、统计分析和仿真/实车两种模式。
"""

import matplotlib.pyplot as plt
import numpy as np
import time

from pal.products.qcar import QCar, IS_PHYSICAL_QCAR
# from pal.utilities.stream import BasicStream , Timeout
from pal.utilities.scope import MultiScope, Signal
from threading import Thread

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup()

plt.rcParams['font.family'] = 'SimHei'
Signal.maxChunks = 1500
#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : SensorInterfacing Class Setup

class sensorInterfacing():
    def __init__(self, taskRate, specifiedSamples):

        # QCar DAQ参数配置
        self.readRate = int(taskRate) # 采样频率Hz
        self.samples  = int(specifiedSamples) # 采样点数
        self.mode     = "read"

        # QCar对象初始化
        self.myQCar = QCar(frequency = self.readRate, readMode = 1)

        # 传感器数据缓存
        self.accelerometerData = np.zeros((3))
        self.gyroscopeData     = np.zeros((3))
        self.velocityData = np.zeros((self.samples,3))
        self.positionData = np.zeros((self.samples,3))
        self.plotData = np.zeros((self.samples,6))
        self.time = np.zeros((int(specifiedSamples)))

        # 加速度计可视化Scope
        txt = "加速度计数据"
        self.accelPlot = MultiScope(title=txt, rows=1, cols=1, fps=taskRate)
        self.accelPlot.addAxis(row=0, col=0,
                    timeWindow=int(self.samples*(1/self.readRate)),
                    yLabel='m/s^2',
                    xLabel='时间 (s)')
        self.accelPlot.axes[0].attachSignal(name='X轴加速度')
        self.accelPlot.axes[0].attachSignal(name='Y轴加速度')
        self.accelPlot.axes[0].attachSignal(name='Z轴加速度')
        self.accelPlot.axes[0].yLim = (-11, 11)

        # 陀螺仪可视化Scope
        txt = "陀螺仪数据"
        self.gyroPlot = MultiScope(title=txt, rows=1, cols=1, fps=taskRate)
        self.gyroPlot.addAxis(row=0, col=0,
                    timeWindow=int(self.samples*(1/self.readRate)),
                    yLabel='rad/s',
                    xLabel='时间 (s)')
        self.gyroPlot.axes[0].attachSignal(name='X轴角速度')
        self.gyroPlot.axes[0].attachSignal(name='Y轴角速度')
        self.gyroPlot.axes[0].attachSignal(name='Z轴角速度')
        self.gyroPlot.axes[0].yLim = (-2.5, 2.5)

        # 位置显示Scope (X-Y平面)
        txt = "位置数据"
        self.posPlot = MultiScope(title=txt, rows=1, cols=1, fps=taskRate)
        self.posPlot.addAxis(row=0, col=0,
                    timeWindow=int(self.samples*(1/self.readRate)),
                    yLabel='Y (m)',
                    xLabel='X (m)')
        self.posPlot.axes[0].attachSignal(name='Y坐标')


    # Timing function used to send timestamp to sigmoid generator.
    def elapsedTime(self,startTime):
        return time.time()-startTime

    def sensor_read(self):
        t0 = time.time()
        loopTime = self.samples*(1/self.readRate)  # Simulation Time
        print("Sim time: ", loopTime)
        self.timeStamp = 0

        # 用于积分的初始速度和位置
        velocity = np.zeros(3)
        position = np.zeros(3)
        prevTimeStamp = 0

        # Run the data logging only for the length of time specified
        i = 0
        while i < self.samples:
            if self.mode == "read":
                if self.timeStamp < int(loopTime/2):
                    self.myQCar.write(throttle=0.075, steering=0.5)
                if self.timeStamp >= int(loopTime/2):
                    self.myQCar.write(throttle=0.075, steering=-0.5)

            # 更新采样时间
            self.timeStamp = self.elapsedTime(t0)
            self.time[i] = self.timeStamp

            # Quanser's implementation for parsing out the IMU information
            self.myQCar.read()  # Read sensor data from the QCar
            self.accelerometerData = self.myQCar.accelerometer
            self.gyroscopeData = self.myQCar.gyroscope

            self.plotData[i,0:3] = self.accelerometerData
            self.plotData[i,3:6] = self.gyroscopeData

            # 数据传递给加速度计和陀螺仪显示
            self.accelPlot.axes[0].sample(self.timeStamp,
                                  [self.accelerometerData[0],
                                   self.accelerometerData[1],
                                   self.accelerometerData[2]].copy())
            self.gyroPlot.axes[0].sample(self.timeStamp,
                                 [self.gyroscopeData[0],
                                  self.gyroscopeData[1],
                                  self.gyroscopeData[2]].copy())

            # region: 学生实现部分——加速度积分成为速度和位置
            # 计算时间间隔 dt
            if i == 0:
                dt = 1/self.readRate  # 或直接取 1/Hz
            else:
                dt = self.timeStamp - prevTimeStamp
            prevTimeStamp = self.timeStamp

            # 欧拉积分：更新速度和位置
            velocity = velocity + self.accelerometerData * dt
            position = position + velocity * dt

            self.velocityData[i] = velocity
            self.positionData[i] = position

            # 实时显示位置（X为横坐标，Y为纵坐标，忽略Z）
            self.posPlot.axes[0].sample(position[0], [position[1]])

            i += 1
            # endregion

        self.myQCar.write(throttle=0, steering=0)

    def sensor_stats(self):

        # ========== 统计特性分析 ==========
        mu = np.zeros(6)
        var = np.zeros(6)
        # 计算均值和方差
        for i in range(6):
            mu[i] = self.plotData[:,i].mean()
            var[i] = self.plotData[:,i].var()
        print("均值: ", mu)
        print("方差: ", var)

        # ========== 数据绘图 ==========
        # 陀螺仪z轴数据示例
        fig1 = plt.figure()
        plt.plot(self.time, self.plotData[:,5])
        plt.title("陀螺仪z轴数据")
        plt.xlabel('时间 (s)')
        plt.ylabel(r'$\omega_z$ (rad/s)')
        plt.grid()

        fig2 = plt.figure()
        plt.hist(self.plotData[:,5], bins=9)
        plt.title("陀螺仪z轴直方图")
        plt.xlabel(r'$\omega_z$ (rad/s)')
        plt.ylabel('采样数')
        plt.grid()
        plt.show()

        # 传感器特性可视化
        fig1, ax1 = plt.subplots(2,3)
        fig2, ax2 = plt.subplots(2,3)
        for i in range(3):
            # 各信号随时间变化
            ax1[0,i].plot(self.time, self.plotData[:,i])
            ax1[1,i].plot(self.time, self.plotData[:,i+3])
            # 各信号直方图
            ax2[0,i].hist(self.plotData[:,i], bins=9)
            ax2[1,i].hist(self.plotData[:,i+3], bins=9)
        ax1[0,0].set(xlabel='时间 (s)', ylabel=r'$a_x$ ($m/s^2$)')
        ax1[0,1].set(xlabel='时间 (s)', ylabel=r'$a_y$ ($m/s^2$)')
        ax1[0,2].set(xlabel='时间 (s)', ylabel=r'$a_z$ ($m/s^2$)')
        ax1[1,0].set(xlabel='时间 (s)', ylabel=r'$\omega_x$ (rad/s)')
        ax1[1,1].set(xlabel='时间 (s)', ylabel=r'$\omega_y$ (rad/s)')
        ax1[1,2].set(xlabel='时间 (s)', ylabel=r'$\omega_z$ (rad/s)')
        ax2[0,0].set(xlabel=r'$a_x$ ($m/s^2$)', ylabel='采样数')
        ax2[0,1].set(xlabel=r'$a_y$ ($m/s^2$)', ylabel='采样数')
        ax2[0,2].set(xlabel=r'$a_z$ ($m/s^2$)', ylabel='采样数')
        ax2[1,0].set(xlabel=r'$\omega_x$ (rad/s)', ylabel='采样数')
        ax2[1,1].set(xlabel=r'$\omega_y$ (rad/s)', ylabel='采样数')
        ax2[1,2].set(xlabel=r'$\omega_z$ (rad/s)', ylabel='采样数')
        fig1.tight_layout()
        fig2.tight_layout()
        plt.show()

    def start(self, mode="read"):

        if mode == "read":
            mainThread = Thread(target=self.sensor_read)
            mainThread.start()

            while mainThread.is_alive():
                self.accelPlot.refresh()
                self.gyroPlot.refresh()
                self.posPlot.refresh()

            time.sleep(5)

        if mode == "sensor_stats":
            self.mode = mode
            mainThread = Thread(target=self.sensor_read)
            mainThread.start()

            while mainThread.is_alive():
                self.accelPlot.refresh()
                self.gyroPlot.refresh()
                self.posPlot.refresh()

            self.sensor_stats()

#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Main Loop
def main():
    try:
        '''
        输入参数：
        taskRate = 采样频率（Hz）
        specifiedSamples = 采样点数
        '''
    # ========== 学生输入参数 ===========
        sensorLab = sensorInterfacing(taskRate = 120,
                    specifiedSamples = 600)

        '''
        可选实验内容：
        - read         ：IMU数据采集与可视化
        - sensor_stats ：IMU噪声统计分析
        '''
        sensorMode = "read" # "read" 或 "sensor_stats"
        sensorLab.start(mode = sensorMode)
    finally:
        if not IS_PHYSICAL_QCAR:
            qlabs_setup.terminate()

        input('实验结束，按任意键退出...')


#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Run
if __name__ == '__main__':
    main()
#endregion
