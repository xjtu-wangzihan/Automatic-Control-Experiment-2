from pal.products.qcar import QCar, QCarRealSense
import time
import numpy as np 
import cv2

# --- 控制参数：速度与避障触发 ---
BASE_SPEED = 0.15        # 巡航油门（0~1）
AVOID_STEER = 0.15       # 避障转角幅度，越大转向越激进
LOOK_AHEAD_DIST = 1.5    # 障碍触发阈值（米）

# --- 状态机各阶段持续时间（秒）---
T_OUT = 0.53             # STEP1: 向左斜出
T_FIX = 0.53             # STEP2: 反向回正并与障碍平行
T_PASS = 1.8             # STEP3: 直行通过障碍
T_IN = 0.47              # STEP4: 向右斜回原车道
T_ALIGN = 0.47           # STEP5: 左打回正，恢复原朝向

# --- 设备初始化 ---
myCar = QCar()
print("正在连接摄像头...")
myCam = QCarRealSense(mode='RGB, Depth')

# RealSense 初始帧可能不稳定，先短暂预热可降低首帧阻塞概率
print("等待 RealSense 预热...")
time.sleep(1) 

startTime = time.time()
state = 'FORWARD'
state_timer = 0

def get_t(): 
    # 返回程序运行到当前的相对时间（秒）
    return time.time() - startTime

# --- 车道线检测参数 ---
# HSV 颜色阈值（用于左右轨迹线可视化）
YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([40, 255, 255])
GREEN_LOWER = np.array([35, 50, 50])
GREEN_UPPER = np.array([85, 255, 255])

# 只处理图像底部区域，减少远处干扰并提升速度
ROI_Y_START = 0.4

def detect_lane_lines(rgb_img, color='yellow'):
    """检测并绘制指定颜色车道线。

    参数:
        rgb_img: BGR 图像
        color: 'yellow' 或 'green'
    返回:
        绘制车道线后的图像
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    
    if color == 'yellow':
        mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    else:  # green
        mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    
    # ROI: 仅保留底部区域
    h, w = mask.shape
    roi_mask = mask[int(h*ROI_Y_START):, :]
    
    # 边缘检测
    edges = cv2.Canny(roi_mask, 50, 150)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=30)
    
    # 将检测结果绘制回原图
    result_img = rgb_img.copy()
    line_color = (0, 255, 255) if color == 'yellow' else (0, 255, 0)  # 黄色或绿色
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # ROI 内坐标还原为原图坐标
            y1 += int(h * ROI_Y_START)
            y2 += int(h * ROI_Y_START)
            cv2.line(result_img, (x1, y1), (x2, y2), line_color, 3)
    
    return result_img

def get_warning_color(distance):
    """根据障碍距离返回预警颜色（BGR）。"""
    if distance < 1.0:
        return (0, 0, 255)  # 红色 - 危险
    elif distance < 1.5:
        return (0, 255, 255)  # 黄色 - 警告
    else:
        return (0, 255, 0)  # 绿色 - 安全

try:
    while get_t() < 120:
        # 1) 读取 RGB 与深度帧
        myCam.read_RGB()
        myCam.read_depth(dataMode='PX')
        
        rgb = myCam.imageBufferRGB
        depth = myCam.imageBufferDepthPX
        
        # 2) 红色障碍分割
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        # 这里只检测低色相红色区间（0~10）
        mask = cv2.inRange(hsv, np.array([0,150,50]), np.array([10,255,255]))
        
        obs_detected = False
        dist = 999.0
        obs_center = None
        obs_bbox = None

        if np.any(mask):
            M = cv2.moments(mask)
            if M["m00"] > 500:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                obs_center = (cX, cY)
                
                # 深度像素值映射到近似米制距离
                val = depth[cY, cX][0] if isinstance(depth[cY, cX], np.ndarray) else depth[cY, cX]
                dist = (val * 39.0 / 255.0) / 10.0
                
                # 障碍在视野中部且距离进入阈值时触发避障
                if dist < LOOK_AHEAD_DIST and 200 < cX < 400:
                    obs_detected = True
                
                # 计算障碍物边界框
                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    obs_bbox = (x_min, y_min, x_max, y_max)

        # 3) 避障状态机：FORWARD -> STEP1..STEP5 -> FORWARD
        curr_t = get_t()
        throttle = BASE_SPEED
        steering = 0.0

        if state == 'FORWARD':
            if obs_detected:
                print(f"!!! 发现障碍: {dist:.2f}m - 立即左转")
                state = 'STEP1'; state_timer = curr_t
        
        elif state == 'STEP1': # 斜出
            steering = -AVOID_STEER
            if curr_t - state_timer > T_OUT: state = 'STEP2'; state_timer = curr_t
        
        elif state == 'STEP2': # 回正平行
            steering = AVOID_STEER
            if curr_t - state_timer > T_FIX: state = 'STEP3'; state_timer = curr_t
        
        elif state == 'STEP3': # 直行通过
            steering = 0.0
            if curr_t - state_timer > T_PASS: state = 'STEP4'; state_timer = curr_t
        
        elif state == 'STEP4': # 斜回
            steering = AVOID_STEER
            if curr_t - state_timer > T_IN: state = 'STEP5'; state_timer = curr_t
        
        elif state == 'STEP5': # 回正原方向
            steering = -AVOID_STEER
            if curr_t - state_timer > T_ALIGN: state = 'FORWARD'

        # 4) 写入纵向/横向控制指令
        myCar.read_write_std(throttle, steering)

        # 5) 可视化（车道线 + 障碍框 + 状态文字）
        display_img = rgb.copy()
        
        # 避障路径阶段用绿色，回归原车道阶段用黄色
        if state == 'FORWARD' or state == 'STEP1' or state == 'STEP5':
            lane_color = 'yellow'
        else:  # STEP2, STEP3, STEP4
            lane_color = 'green'
        
        # 绘制车道线
        display_img = detect_lane_lines(display_img, lane_color)
        
        # 绘制障碍物框选和预警
        if obs_bbox is not None and obs_center is not None:
            x_min, y_min, x_max, y_max = obs_bbox
            warning_color = get_warning_color(dist)
            
            # 绘制边界框
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), warning_color, 3)
            
            # 绘制中心点
            cv2.circle(display_img, obs_center, 5, warning_color, -1)
            
            # 显示距离和预警文字
            warning_text = f"{dist:.2f}m"
            if dist < 1.0:
                status_text = "DANGER"
            elif dist < 1.5:
                status_text = "WARNING"
            else:
                status_text = "SAFE"
            
            # 绘制文字背景
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(display_img, (x_min, y_min - 30), (x_min + text_size[0] + 10, y_min), warning_color, -1)
            cv2.putText(display_img, warning_text, (x_min + 5, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 在框上方显示状态
            cv2.putText(display_img, status_text, (x_min, y_min - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warning_color, 2)
        
        # 显示状态信息
        cv2.putText(display_img, f"State: {state} Dist: {dist:.1f}m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('RGB', display_img)
        if cv2.waitKey(1) == 27: break

finally:
    # 无论如何退出，都先停车再释放硬件
    myCar.read_write_std(0, 0)
    myCar.terminate()
    myCam.terminate()
