# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

"""
QCar2_Speed_Control.py
本程序用于QCar2小车纵向速度控制实验，演示如何基于编码器反馈实现速度闭环控制（PI控制），并通过MultiScope实时可视化速度、误差和控制信号。
支持仿真与实车两种模式，包含安全中断、线程控制等功能。
"""
import signal
import numpy as np
from threading import Thread
import time

from pal.products.qcar import QCar, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import *


#================ 实验参数配置 ================
# ===== 时间参数 =====
# tf: 实验总时长（秒）
# startDelay: 启动延迟，滤波器稳定用（秒）
# controllerUpdateRate: 控制循环频率（Hz），不建议超过500
tf = 10
startDelay = 1
controllerUpdateRate = 100
sampleTime = 1/controllerUpdateRate

# ===== 速度控制器参数 =====
# v_ref: 期望速度（m/s）
# K_p: 比例增益
# K_i: 积分增益
v_ref = 0.9
K_p = 1
K_i = 0.5

# 差分器初始化
diff = Calculus().differentiator_variable(sampleTime)
_ = next(diff)
timeStep = sampleTime
#endregion

#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 初始设置

initialPose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    import qlabs_setup
    qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )
    calibrate = False
else:
    calibrate =  'y' in input('是否需要重新标定？(y/n)')

# 用于安全地通过键盘中断实验
global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
#endregion


# 速度PI控制器类
class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3  # 最大油门限制
        self.kp = kp
        self.ki = ki
        self.ei = 0  # 积分项

    # 速度控制器更新
    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += e * dt
        u = self.kp * e + self.ki * self.ei
        u = np.clip(u, -self.maxThrottle, self.maxThrottle)
        return u


# 控制主循环，负责读取编码器、计算速度、PI控制、写入控制指令、刷新可视化
def controlLoop():
    #region : 控制循环初始化
    global KILL_THREAD
    u = 0
    delta = 0
    # 用于将数据采样频率限制为10Hz
    countMax = controllerUpdateRate / 10
    count = 0
    #endregion

    #region : 控制器初始化
    speedController = SpeedController(
        kp=K_p,
        ki=K_i,
    )
    #endregion

    #region : QCar接口初始化
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)

    with qcar:
        t0 = time.time()
        t = 0
        while (t < tf + startDelay) and (not KILL_THREAD):
            #region : 循环计时
            tp = t
            t = time.time() - t0
            dt = t - tp
            #endregion

            #region : 读取传感器并计算速度
            qcar.read()
            encoderCounts = qcar.motorEncoder[0]
            L = (1/720/4)  *  ( (13*19) / (70*37) )  *  1   *  2*np.pi  *  0.0342  *  encoderCounts
            encoderSpeed = diff.send((encoderCounts, timeStep))
            v = (1/720/4)  *  ( (13*19) / (70*37) )  *  1   *  2*np.pi  *  0.0342  *  encoderSpeed
            #endregion

            #region : PI速度控制器
            if t < startDelay:
                u = 0
                delta = 0
            else:
                u = speedController.update(v, v_ref, dt)
                delta = 0
            #endregion

            qcar.write(u, delta)

            #region : 刷新可视化曲线
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay
                # 多通道Scope采样
                speedScope.axes[0].sample(t_plot, [v, v_ref])
                speedScope.axes[1].sample(t_plot, [v_ref-v])
                speedScope.axes[2].sample(t_plot, [u])
                count = 0
            #endregion
            continue
        qcar.read_write_std(throttle=0, steering=0)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 实验主流程
if __name__ == '__main__':

    #region : Scope可视化配置
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
    # 多通道Scope用于监控速度控制
    speedScope = MultiScope(
        rows=3,
        cols=1,
        title='纵向速度控制',
        fps=fps
    )
    speedScope.addAxis(
        row=0,
        col=0,
        timeWindow=tf,
        yLabel='直线速度 [m/s]',
        yLim=(0, 1)
    )
    speedScope.axes[0].attachSignal(name='测量速度', width=2)
    speedScope.axes[0].attachSignal(name='期望速度')
    speedScope.addAxis(
        row=1,
        col=0,
        timeWindow=tf,
        yLabel='速度误差 [m/s]',
        yLim=(-0.5, 0.5)
    )
    speedScope.axes[1].attachSignal()
    speedScope.addAxis(
        row=2,
        col=0,
        timeWindow=tf,
        xLabel='时间 [s]',
        yLabel='PWM信号(油门) [%]',
        yLim=(-0.3, 0.3)
    )
    speedScope.axes[2].attachSignal()

    #region : 启动控制线程并运行实验
    controlThread = Thread(target=controlLoop)
    controlThread.start()
    try:
        while controlThread.is_alive() and (not KILL_THREAD):
            MultiScope.refreshAll()
            time.sleep(0.01)
    finally:
        KILL_THREAD = True
    #endregion
    if not IS_PHYSICAL_QCAR:
        qlabs_setup.terminate()
    input('实验结束，按任意键退出...')
#endregion