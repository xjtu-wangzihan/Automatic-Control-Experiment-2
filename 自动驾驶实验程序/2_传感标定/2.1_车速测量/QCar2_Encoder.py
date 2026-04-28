"""
QCar2_Encoder.py
本程序用于QCar2小车编码器实验，演示如何读取编码器数据、计算里程和速度，并通过MultiScope实时可视化。
支持仿真与实车两种模式，包含安全中断、线程控制等功能。
"""
import signal
import numpy as np
from threading import Thread
import time

from pal.products.qcar import QCar, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import Calculus
import qlabs_setup

#================ 实验参数配置 ================
# ===== 时间参数 =====
# tf: 实验总时长（秒）
# startDelay: 启动延迟，滤波器稳定用（秒）
# controllerUpdateRate: 控制循环频率（Hz），不建议超过500
tf = 15
startDelay = 1
controllerUpdateRate = 100
sampleTime = 1/controllerUpdateRate

#region : 差分器初始化
diff = Calculus().differentiator_variable(sampleTime)
_ = next(diff)  # 预热差分器，丢弃第一次输出的异常值
timeStep = sampleTime
#endregion

#region : 整体初始化设置

initialPose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    qlabs_setup.setup(
        initialPosition=[initialPose[0], initialPose[1], 0],
        initialOrientation=[0, 0, initialPose[2]]
    )
    calibrate = False
else:
    calibrate = 'y' in input('do you want to recalibrate?(y/n)')

# 用于安全地通过键盘中断实验
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
#endregion

# 控制主循环，负责读取编码器、计算速度与里程、写入控制指令、刷新可视化
def controlLoop():
    
    # 用于将数据采样频率限制为10Hz
    countMax = controllerUpdateRate / 10
    count = 0
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)

    with qcar:
        t0 = time.time()
        t = 0
        while (t < tf + startDelay) and (not KILL_THREAD):

            t = time.time() - t0

            # ============== 读取并处理编码器数据 ==============
            qcar.read()
            encoderCounts = qcar.motorEncoder[0]
            # 里程计算公式
            L = (1/720/4) * (13*19)/(70*37) * 1 * 2*np.pi * 0.0342 * encoderCounts
            encoderSpeed = diff.send((encoderCounts, timeStep))
            v = (1/720/4) * (13*19)/(70*37) * 1 * 2*np.pi * 0.0342 * encoderSpeed

            #region : 控制指令写入
            if t < startDelay:
                u = 0
                delta = 0
            else:
                u = 0.1
                delta = 0
            #endregion

            qcar.write(u, delta)

            #region : 刷新可视化曲线
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay
                # 多通道Scope采样
                speedScope.axes[0].sample(t_plot, [encoderCounts])
                speedScope.axes[1].sample(t_plot, [L])
                speedScope.axes[2].sample(t_plot, [encoderSpeed])
                speedScope.axes[3].sample(t_plot, [v])
                count = 0
            #endregion
            continue
        qcar.read_write_std(throttle=0, steering=0)


#region : 实验主流程
if __name__ == '__main__':

    #region : Scope可视化配置
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
    # 多通道Scope用于监控编码器与速度
    speedScope = MultiScope(rows=4,cols=1,title='编码器读取及处理',fps=fps)
    speedScope.addAxis(row=0,col=0,timeWindow=tf,yLabel='编码器读数 [counts]',yLim=(0, 900000))
    speedScope.axes[0].attachSignal(name='编码器读数 [counts]', width=2)
    speedScope.addAxis(row=1,col=0,timeWindow=tf,yLabel='直线里程 [m]',yLim=(0, 10))
    speedScope.axes[1].attachSignal(name='直线里程 [m]', width=2)
    speedScope.addAxis(row=2,col=0,timeWindow=tf,yLabel='编码器速度 [counts/s]',yLim=(0, 100000))
    speedScope.axes[2].attachSignal(name='编码器速度 [counts/s]', width=2)
    speedScope.addAxis(row=3,col=0,timeWindow=tf,xLabel='时间 [s]',yLabel='直线速度 [m/s]',yLim=(-0.5, 3))
    speedScope.axes[3].attachSignal(name='直线速度 [m/s]', width=2)

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