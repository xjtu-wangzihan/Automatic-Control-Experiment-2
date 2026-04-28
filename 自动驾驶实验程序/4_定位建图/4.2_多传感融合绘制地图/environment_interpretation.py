# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

"""
environment_interpretation.py

这是一个基于QCAR的环境感知与建图程序。主要功能包括：
1. 使用激光雷达进行环境扫描
2. 构建并更新栅格地图
3. 实现车辆自主导航
4. 实时显示建图结果

本程序使用占据栅格地图方法进行环境建模，通过多线程实现同时建图和控制。
"""

import numpy as np
from scipy.special import logit, expit  # 用于概率与对数几率转换
from scipy import ndimage  # 用于图像处理
from threading import Thread, Lock  # 用于多线程控制
import time
import pyqtgraph as pg  # 用于实时数据可视化
import signal  # 用于程序终止信号处理

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR, QCAR_CONFIG
from pal.utilities.scope import MultiScope
from pal.utilities.math import find_overlap, wrap_to_2pi
from hal.content.qcar_functions import QCarEKF, QCarDriveController
from hal.products.mats import SDCSRoadMap
#endregion

# ================ Experiment Configuration ================
# ===== 时间参数设置
# - tf: 实验持续时间（秒）
# - startDelay: 系统初始化延时（秒），给滤波器时间达到稳定状态
# - controllerUpdateRate: 控制器更新频率（Hz），不应超过500Hz
tf = 100
startDelay = 1
controllerUpdateRate = 100

# ===== 车辆控制参数
# - enableVehicleControl: 为True时，QCAR将按照指定的路径点序列行驶；
#                        为False时，车辆保持静止
# - v_ref: 期望速度（米/秒）
# - nodeSequence: 路径点序列，用于轨迹生成
enableVehicleControl = True
v_ref = 0.3
nodeSequence = [0, 20, 0]

# ===== 占据栅格地图参数
# - cellWidth: 栅格单元格的边长（米）
# - r_res: 极坐标栅格的距离分辨率（米）
# - r_max: 雷达的最大测量范围（米）
# - p_low: 空闲栅格的概率值（根据雷达扫描判定）
# - p_high: 占据栅格的概率值（根据雷达扫描判定）
cellWidth = 0.02
r_res = 0.05
r_max = 4
p_low = 0.4
p_high = 0.6


#region 系统初始化
# 创建线程锁，用于多线程间的数据同步
lock = Lock()

# 初始化路径规划
roadmap = SDCSRoadMap()                                    # 创建道路地图
waypointSequence = roadmap.generate_path(nodeSequence)     # 生成路径点序列
initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()  # 获取初始位姿
x_hat = initialPose  # 初始化状态估计值
t_hat = 0           # 初始化时间戳

# 根据是否为实物QCAR进行不同的初始化
if not IS_PHYSICAL_QCAR:
    # 仿真环境初始化
    import qlabs_setup
    hQCar = qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )
    calibrate=False
else:
    # 实物环境初始化，询问是否需要重新标定
    calibrate =  'y' in input('是否需要重新标定?(y/n)')

# 设置程序终止信号处理
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

# 初始化GPS和激光雷达
gps = QCarGPS(initialPose=initialPose,calibrate=calibrate,attach_lidar=True)
# 等待传感器初始化完成
while (not KILL_THREAD) and (gps.readGPS() or  gps.readLidar()):
    pass

#endregion

class OccupancyGrid:
    """
    占据栅格地图类
    用于构建和更新环境的栅格化表示
    """
    def __init__(self,
            x_min=-4,      # 地图X轴最小值（米）
            x_max=3,       # 地图X轴最大值（米）
            y_min=-3,      # 地图Y轴最小值（米）
            y_max=6,       # 地图Y轴最大值（米）
            cellWidth=0.02,  # 栅格单元格大小（米）
            r_max=5,       # 雷达最大量程（米）
            r_res=0.02,    # 距离分辨率（米）
            p_low=0.4,     # 空闲栅格概率
            p_high=0.6     # 占据栅格概率
        ):

        #region 定义概率值及其对数几率形式
        self.p_low = p_low          # 空闲栅格概率
        self.p_prior = 0.5          # 先验概率（未知状态）
        self.p_high = p_high        # 占据栅格概率
        self.p_sat = 0.001          # 饱和概率（防止概率值达到0或1）

        # 将概率转换为对数几率形式，便于数值计算
        self.l_low = logit(self.p_low)
        self.l_prior = logit(self.p_prior)
        self.l_high = logit(self.p_high)
        self.l_min = logit(self.p_sat)
        self.l_max = logit(1-self.p_sat)
        #endregion

        self.init_polar_grid(r_max, r_res)

        self.init_world_map(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            cellWidth=cellWidth
        )
        self.init_patch()

    # ==============  极坐标栅格初始化 ====================
    def init_polar_grid(self, r_max, r_res):
        # 配置极坐标栅格参数
        fov = 2*np.pi                # 视场角（360度）
        self.phiRes = 1 * np.pi/180  # 角度分辨率（1度）
        self.r_max = r_max           # 最大测量距离
        self.r_res = r_res           # 距离分辨率

        # 计算极坐标patch的尺寸
        self.mPolarPatch = np.int_(np.ceil(fov / self.phiRes))        # 角度方向的栅格数
        self.nPolarPatch = np.int_(np.floor(self.r_max/self.r_res))   # 距离方向的栅格数

        # 创建极坐标栅格地图，初始化为0
        self.polarPatch = np.zeros(
            shape = (self.mPolarPatch, self.nPolarPatch),
            dtype = np.float32
        )

    def update_polar_grid(self, r):
        """
        更新极坐标栅格地图
        参数:
        r: 激光雷达的距离测量值列表，长度为self.mPolarPatch
           每个测量值之间的角度间隔为self.phiRes弧度
        """
        # 将距离值转换为栅格索引
        r = np.int_(np.round(r / self.r_res))
        
        # 遍历每个角度的测量值
        for i in range(self.mPolarPatch):
            if r[i] > 0:  # 有效测量值
                # 将测量点之前的栅格标记为空闲
                self.polarPatch[i,:r[i]] = self.l_low
                # 将测量点处的栅格标记为占据
                self.polarPatch[i,r[i]:r[i]+1] = self.l_high
                # 将测量点之后的栅格保持为未知状态
                self.polarPatch[i,r[i]+1:] = self.l_prior
            else:  # 无效测量值，保持未知状态
                self.polarPatch[i,:] = self.l_prior
        


    # ==============  栅格插值部分 ====================
    def init_patch(self):
        """初始化局部地图patch
        创建一个正方形的局部地图，用于存储从极坐标转换到笛卡尔坐标系的栅格数据
        """
        # 计算patch的尺寸，确保能覆盖最大测量范围
        self.nPatch = np.int_(2*np.ceil(self.r_max/self.cellWidth) + 1)
        # 创建零矩阵作为patch
        self.patch = np.zeros(
            shape = (self.nPatch, self.nPatch),
            dtype = np.float32
        )

    def generate_patch(self, th):
        """生成局部地图patch
        将极坐标形式的测量数据转换为笛卡尔坐标系下的栅格地图
        参数:
        th: 当前车辆朝向角（弧度）
        """
        # 计算patch中心点坐标
        cx = (self.nPatch * self.cellWidth) / 2
        cy = cx

        # 创建笛卡尔坐标网格
        x = np.linspace(-cx, cx, self.nPatch)
        y = np.linspace(-cy, cy, self.nPatch)
        xv, yv = np.meshgrid(x, y)

        # 计算每个栅格点对应的极坐标值
        rPatch = (
            np.sqrt(np.square(xv) + np.square(yv)) / self.r_res  # 距离值
        )
        phiPatch = (
            wrap_to_2pi(np.arctan2(yv, xv) + th) / self.phiRes   # 角度值
        )
        
        # 使用双线性插值将极坐标数据映射到笛卡尔坐标系
        ndimage.map_coordinates(
            input=self.polarPatch,
            coordinates=[phiPatch, rPatch],
            output=self.patch
        )
       

    # ==============  全局占据栅格地图更新  ====================
    def init_world_map(self,
            x_min = -4,     # 地图X轴最小值（米）
            x_max = 3,      # 地图X轴最大值（米）
            y_min = -3,     # 地图Y轴最小值（米）
            y_max = 6,      # 地图Y轴最大值（米）
            cellWidth=0.02  # 栅格大小（米）
        ):
        """初始化全局栅格地图"""
        # 保存地图参数
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cellWidth = cellWidth
        
        # 计算地图尺寸
        self.xLength = x_max - x_min
        self.yLength = y_max - y_min
        
        # 计算栅格数量
        self.m = np.int_(np.ceil(self.yLength/self.cellWidth))  # 行数
        self.n = np.int_(np.ceil(self.xLength/self.cellWidth))  # 列数

        # 创建并初始化地图，所有栅格设置为先验概率（未知状态）
        self.map = np.full(
            shape = (self.m, self.n),
            fill_value = self.l_prior,
            dtype = np.float32
        )

    def xy_to_ij(self, x, y):
        """将世界坐标转换为栅格索引
        参数:
        x, y: 世界坐标系中的位置（米）
        返回:
        i, j: 对应的栅格索引（行，列）
        """
        i = np.int_(np.round( (self.y_max - y) / self.cellWidth ))
        j = np.int_(np.round( (x - self.x_min) / self.cellWidth ))
        return i, j

    def updateMap(self, x, y, th, angles, distances):
        """更新全局占据栅格地图
        
        参数:
        x, y: 车辆当前位置（米）
        th: 车辆朝向角（弧度）
        angles: 激光雷达扫描角度列表
        distances: 激光雷达距离测量值列表
        """
        # 更新极坐标栅格地图
        self.update_polar_grid(distances)

        # 生成局部笛卡尔坐标栅格地图
        self.generate_patch(th)

        # 计算当前位置对应的栅格索引
        iy, jx = self.xy_to_ij(x, y)

        # 计算局部地图在全局地图中的位置
        iTop = np.int_(iy - np.round((self.nPatch-1)/2))   # 顶部边界
        jLeft = np.int_(jx - np.round((self.nPatch-1)/2))  # 左侧边界

        # 计算局部地图和全局地图的重叠区域
        mapSlice, patchSlice = find_overlap(
            self.map,
            self.patch,
            iTop,
            jLeft
        )

        # 使用对数几率更新法则更新重叠区域的栅格状态
        # 并确保概率值不会达到0或1（使用clip限制在有效范围内）
        self.map[mapSlice] = np.clip(
            (self.map[mapSlice] + self.patch[patchSlice]),
            self.l_min,
            self.l_max
        )


def controlLoop():
    """车辆控制线程
    实现车辆状态估计和轨迹跟踪控制
    """
    #region 控制循环初始化
    global KILL_THREAD, x_hat, t_hat
    u = 0          # 初始化速度控制量
    delta = 0      # 初始化转向控制量
    # 限制数据采样频率为10Hz
    countMax = controllerUpdateRate / 10
    count = 0
    #endregion

    #region 设置可视化箭头标记
    # 创建箭头1用于显示局部地图中的车辆朝向
    arrow1 = pg.ArrowItem(
        angle=180,
        tipAngle=60,
        headLen=10,
        tailLen=10,
        tailWidth=5,
        pen={'color': 'w', 'width': 1},
        brush='r'
    )
    arrow1.setPos(0,0)
    scope.axes[1].plot.addItem(arrow1)

    # 创建箭头2用于显示全局地图中的车辆朝向
    arrow2 = pg.ArrowItem(
        angle=180,
        tipAngle=60,
        headLen=10,
        tailLen=10,
        tailWidth=5,
        pen={'color': 'w', 'width': 1},
        brush='r'
    )
    scope.axes[2].plot.addItem(arrow2)
    #endregion

    #region QCar interface setup
    with lock:
        ekf = QCarEKF(x_0=x_hat)
    driveController = QCarDriveController(waypointSequence, cyclic=False)

    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    #endregion

    with qcar:
        t0 = time.time()
        t = 0
        while (t < tf+startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion

            #region : Update QCar state estimates and drive controller
            qcar.read()
            if gps.readGPS():
                y_gps = np.array([
                    gps.position[0],
                    gps.position[1],
                    gps.orientation[2]
                ])
                ekf.update(
                    [qcar.motorTach, delta],
                    dt,
                    y_gps,
                    qcar.gyroscope[2],
                )
            else:
                ekf.update(
                    [qcar.motorTach, delta],
                    dt,
                    None,
                    qcar.gyroscope[2],
                )
            with lock:
                t_hat = time.time()
                x_hat = ekf.x_hat[:]

            x = ekf.x_hat[0, 0]
            y = ekf.x_hat[1, 0]
            v = qcar.motorTach
            th = ekf.x_hat[2, 0]
            p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2

            if t < startDelay or (not enableVehicleControl):
                u = 0
                delta = 0
            else:
                u, delta = driveController.update(p, th, v, v_ref, dt)
            qcar.write(u, delta)
            #endregion

            #region : Update Scopes
            count += 1
            if count >= countMax and t > startDelay:
                scope.axes[2].sample(t, [[p[0], p[1]]])
                arrow1.setStyle(angle=180-th*180/np.pi)
                arrow2.setPos(p[0],p[1])
                arrow2.setStyle(angle=180-th*180/np.pi)

                count = 0
            #endregion

            if driveController.steeringController.pathComplete:
                return
            continue
        with lock:
            print('Control thread terminated')


def mappingLoop():
    """地图构建线程
    实现实时激光雷达数据处理和栅格地图更新
    """
    global KILL_THREAD, x_hat, t_hat

    # 创建占据栅格地图对象
    og = OccupancyGrid(
        cellWidth=cellWidth,  # 栅格大小
        r_res=r_res,         # 距离分辨率
        r_max=r_max,         # 最大测量范围
        p_low=p_low,         # 空闲栅格概率
        p_high=p_high        # 占据栅格概率
    )

    #region 配置显示窗口
    # 配置极坐标视图显示
    scope.axes[0].images[0].rotation = 90  # 旋转90度以便于观察
    scope.axes[0].images[0].scale = (og.r_res, -og.phiRes*180/np.pi)  # 设置比例尺
    scope.axes[0].images[0].offset = (0, 0)  # 设置原点位置
    scope.axes[0].images[0].levels = (0, 1)  # 设置显示范围

    # 配置局部地图视图显示
    scope.axes[1].images[0].scale = (og.r_res, -og.r_res)  # 设置比例尺
    scope.axes[1].images[0].offset = (-og.nPatch/2, -og.nPatch/2)  # 居中显示
    scope.axes[1].images[0].levels = (0, 1)  # 设置显示范围

    # 配置全局地图视图显示
    scope.axes[2].images[0].scale = (og.cellWidth, -og.cellWidth)  # 设置比例尺
    scope.axes[2].images[0].offset = (  # 设置地图原点位置
        og.x_min/og.cellWidth,
        -og.y_max/og.cellWidth
    )
    scope.axes[2].images[0].levels = (0, 1)  # 设置显示范围
    #endregion

    t0 = time.time()
    while time.time()-t0 < startDelay:
        gps.readLidar()

    while (not KILL_THREAD):
        #region Get latest pose estimate
        with lock:
            t = t_hat
            x = x_hat[0,0]
            y = x_hat[1,0]
            th = x_hat[2,0]

        x += 0.125 * np.cos(th)
        y += 0.125 * np.sin(th)
        #endregion

        #Read from Lidar and Update Occupancy Grid

        gps.readLidar()

        if gps.scanTime < t and QCAR_CONFIG['cartype'] == 1:
            continue

        og.updateMap(x, y, th, gps.angles,gps.distances)

        scope.axes[0].images[0].setImage(image=expit(og.polarPatch))
        scope.axes[1].images[0].setImage(image=expit(og.patch))
        scope.axes[2].images[0].setImage(image=expit(og.map))

    with lock:
        print('Mapping thread terminated')


#region : 实验设置与运行
if __name__ == '__main__':

    #region : 配置显示窗口
    # 根据是否为实物QCAR设置不同的刷新率
    if IS_PHYSICAL_QCAR:
        fps = 10    # 实物模式下使用较低帧率
    else:
        fps = 30    # 仿真模式下使用较高帧率

    # 创建多窗口显示界面
    scope = MultiScope(
        rows=2,
        cols=2,
        title='环境感知与建图',
        fps=fps
    )

    # 配置极坐标视图（显示激光雷达原始数据）
    scope.addXYAxis(
        row=0,
        col=0,
        xLabel='角度 [度]',
        yLabel='距离 [米]'
    )
    scope.axes[0].attachImage()

    # 配置局部地图视图（显示当前局部栅格地图）
    scope.addXYAxis(
        row=1,
        col=0,
        xLabel='X轴位置 [米]',
        yLabel='Y轴位置 [米]'
    )
    scope.axes[1].attachImage()

    # 配置全局地图视图（显示完整栅格地图和车辆轨迹）
    scope.addXYAxis(
        row=0,
        col=1,
        rowSpan=2,
        xLabel='X轴位置 [米]',
        yLabel='Y轴位置 [米]',
        xLim=(-4, 3),
        yLim=(-2, 6)
    )
    scope.axes[2].attachSignal(name='实际轨迹', width=2, style='--.')
    scope.axes[2].attachImage()

    # 添加参考路径显示
    referencePath = pg.PlotDataItem(
        pen={'color': (85,168,104), 'width': 2},
        name='参考路径'
    )
    referencePath.setData(waypointSequence[0, :], waypointSequence[1, :])
    scope.axes[2].plot.addItem(referencePath)
    #endregion

    #region : 启动线程并运行实验
    # 创建并启动控制和建图线程
    controlThread = Thread(target=controlLoop)    # 车辆控制线程
    mappingThread = Thread(target=mappingLoop)    # 地图构建线程

    controlThread.start()
    mappingThread.start()

    try:
        # 主循环：刷新显示界面
        while controlThread.is_alive() and mappingThread.is_alive():
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        # 程序终止处理
        KILL_THREAD = True  # 设置终止标志
        controlThread.join()  # 等待控制线程结束
        mappingThread.join()  # 等待建图线程结束
        gps.terminate()      # 关闭GPS和激光雷达
    #endregion

    # 如果是仿真模式，关闭仿真环境
    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()

    input('实验完成，按任意键退出...')
#endregion
