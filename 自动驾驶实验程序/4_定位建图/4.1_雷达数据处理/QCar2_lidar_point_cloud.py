"""
QCar2激光雷达点云显示程序

本程序通过处理激光雷达数据实现以下功能：
1. 实时采集并显示QCar周围环境的激光雷达点云数据
2. 实现两种可视化方式：
   - 极坐标显示：展示雷达原始扫描数据，并用概率方式表示空间占用情况
   - 笛卡尔坐标显示：将点云数据转换为X-Y平面上的散点图
3. 通过实时更新实现动态环境感知
"""

from pal.products.qcar import QCarLidar
from pal.utilities.math import *
from pal.utilities.scope import MultiScope
import pyqtgraph as pg
from scipy.special import logit, expit
import time
import numpy as np

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## 程序运行时间控制参数
# 记录程序开始时间
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

sampleRate = 30          # 采样频率(Hz)
sampleTime = 1/sampleRate    # 采样周期(秒)
simulationTime = 30.0    # 程序运行总时长(秒)
print('采样周期: ', sampleTime)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## Additional parameters and buffers
# 雷达扫描信号画图参数
display_range = 4.0  # 极坐标显示范围(米)，仅用于显示
r_res = 0.02      # 距离分辨率(米)
phiRes = 1 * np.pi/180  # 角度分辨率(弧度)

# 图像显示灰度参数
p_low = 0.4    # 扫描被遮挡区域灰度值
p_high = 0.6   # 扫描未被遮挡区域灰度值
p_prior = 0.5  # 无数据区域灰度值

# 转换为log odds形式便于凸显
l_low = logit(p_low)
l_high = logit(p_high)
l_prior = logit(p_prior)

# 极坐标网格参数 
mPolarPatch = np.int_(np.ceil(2*np.pi / phiRes))  # 角度方向网格数
nPolarPatch = np.int_(np.floor(display_range/r_res))  # 距离方向网格数

# 创建极坐标网格
polarPatch = np.full((mPolarPatch, nPolarPatch), l_prior, dtype=np.float32)


# 激光雷达配置
numMeasurements = 1000     # 每次扫描的采样点数
lidarMeasurementMode = 2   # 测距模式：2表示标准模式
lidarInterpolationMode = 0 # 插值模式：0表示不使用插值

# 初始化激光雷达对象
myLidar = QCarLidar(
    numMeasurements=numMeasurements,
    rangingDistanceMode=lidarMeasurementMode,
    interpolationMode=lidarInterpolationMode
)


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
## 配置可视化界面
# 创建1行2列的显示窗口
scope = MultiScope(
    rows=1,
    cols=2,
    title='激光雷达点云显示',
    fps=30  # 显示刷新率30Hz
)

# 设置窗口大小
scope.graphicsLayoutWidget.resize(1200, 600)  # 宽1200像素，高600像素

# 配置极坐标显示区域
scope.addXYAxis(
    row=0,
    col=0,
    xLabel='Angle [deg]',
    yLabel='Range [m]'
)
scope.axes[0].attachImage()

# 设置极坐标图像显示参数
scope.axes[0].images[0].rotation = 90
scope.axes[0].images[0].scale = (r_res, -phiRes*180/np.pi)
scope.axes[0].images[0].offset = (0, 0)
scope.axes[0].images[0].levels = (0, 1)

# 配置笛卡尔坐标显示区域
scope.addXYAxis(
    row=0,
    col=1,
    xLabel='X [m]',
    yLabel='Y [m]',
    xLim=(-10, 10),
    yLim=(-10, 10)
)
scope.axes[1].plot.showGrid(True, True)
# 设置网格线样式
scope.axes[1].plot.getAxis('bottom').setStyle(tickLength=10)
scope.axes[1].plot.getAxis('left').setStyle(tickLength=10)
scope.axes[1].plot.showGrid(x=True, y=True, alpha=0.5)

# 添加散点图
scatter = pg.ScatterPlotItem(
    size=1.5,  # 增大点的大小
    brush=pg.mkBrush(255, 255, 255, 180),  # 增加不透明度
    pen=pg.mkPen(None)  # 移除点的边框
)
scope.axes[1].plot.addItem(scatter)

## 主循环
try:
    while elapsed_time() < simulationTime:
        # 记录本次循环开始时间
        start = time.time()

        # 读取激光雷达数据
        myLidar.read()

        # 获取距离和角度数据并转换为numpy数组
        distances = np.array(myLidar.distances)  # 距离数据(米)
        angles = np.array(myLidar.angles)        # 角度数据(弧度)
        # 坐标系转换：
        # 1. 雷达坐标系需要顺时针旋转90度到车身坐标系
        # 2. 角度方向从逆时针改为顺时针以匹配显示需求
        angles = wrap_to_2pi(2.5*np.pi-angles)
        
        # 更新极坐标网格（仅显示范围内的数据）
        polarPatch.fill(l_prior)  # 重置为无数据时的灰度
        for i in range(mPolarPatch):
            if distances[i] > 0:
                # 计算网格索引
                r_idx = np.int_(np.round(min(distances[i], display_range) / r_res))
                if r_idx < nPolarPatch:
                    # 标记路径上的空闲区域
                    polarPatch[i, :r_idx] = l_low
                    # 标记探测到物体的位置
                    if r_idx > 0:  # 确保有效的探测点
                        polarPatch[i, r_idx-1:r_idx+1] = l_high  # 加宽探测点以增强显示效果

        # 更新极坐标显示
        scope.axes[0].images[0].setImage(image=expit(polarPatch))
        
        # 计算点云的笛卡尔坐标（显示所有有效点）
        valid_idx = distances > 0  # 只过滤掉无效读数
        x = distances[valid_idx] * np.cos(angles[valid_idx])
        y = distances[valid_idx] * np.sin(angles[valid_idx])
        
        # 更新点云显示
        scatter.setData(x=x, y=y)
        
        # End timing this iteration
        end = time.time()

        # Calculate the computation time, and the time that the thread should pause/sleep for
        computationTime = end - start
        sleepTime = sampleTime - (computationTime % sampleTime)

        # 刷新界面显示
        MultiScope.refreshAll()
        time.sleep(max(0, sleepTime))


except KeyboardInterrupt:
    print("User interrupted!")

finally:
    # Terminate the LIDAR object
    myLidar.terminate()
    input('按任意键退出...')
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --