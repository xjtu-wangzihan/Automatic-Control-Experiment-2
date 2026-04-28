from pal.products.qcar import QCar, QCarRealSense
import time
import numpy as np 
import cv2
import signal

# --- 极速响应参数 ---
BASE_SPEED = 0.15    # 前进速度
AVOID_STEER = 0.15        # 转向角
LOOK_AHEAD_DIST = 1.4    # 前视距离阈值 (米)

# --- RGB/Depth 视场角参数，用于坐标对齐 ---
DEPTH_H_FOV = 86.0
DEPTH_V_FOV = 57.0
RGB_H_FOV = 69.4
RGB_V_FOV = 42.5

# --- 状态机时间 (秒) ---
T_OUT = 0.852; 
T_FIX = 0.852; 
T_PASS = 0.05; 

# --- 初始化 ---
myCar = QCar()
print("正在连接摄像头...")
myCam = QCarRealSense(mode='RGB, Depth')

# 解决 read_depth 挂起的关键：强制等待 3 秒预热
print("等待 RealSense 预热...")
time.sleep(3) 

startTime = time.time()
state = 'FORWARD'
# state_timer = 0
stop_requested = False
avoid_dir = -1
# 与 qlabs_setup 保持一致：初始车位在左车道
current_lane = 'LEFT'


def handle_sigint(*args):
    global stop_requested
    stop_requested = True
    print("检测到 Ctrl+C,安全停止实验...")


signal.signal(signal.SIGINT, handle_sigint)


def get_time(): 
    return time.time() - startTime


def get_warning_color(distance):
    if distance < 1.0:
        return (0, 0, 255)      # 红色: 危险
    elif distance < 1.5:
        return (0, 255, 255)    # 黄色: 警告
    else:
        return (0, 255, 0)      # 绿色: 安全


def rgb_to_depth_pixel(cX, cY, rgb_shape, depth_shape):
    # 以图像中心为原点，用视场角比例把 RGB 像素映射到 Depth 像素。
    rgb_h, rgb_w = rgb_shape[:2]
    depth_h, depth_w = depth_shape[:2]

    x_rel = cX - (rgb_w // 2)
    y_rel = cY - (rgb_h // 2)

    h_scale = RGB_H_FOV / DEPTH_H_FOV
    v_scale = RGB_V_FOV / DEPTH_V_FOV

    depth_x = int(round(x_rel * h_scale + (depth_w // 2)))
    depth_y = int(round(y_rel * v_scale + (depth_h // 2)))

    depth_x = int(np.clip(depth_x, 0, depth_w - 1))
    depth_y = int(np.clip(depth_y, 0, depth_h - 1))
    return depth_x, depth_y

try:
    while get_time() < 60 and not stop_requested:
        # 1. 快速读取数据
        myCam.read_RGB()
        myCam.read_depth(dataMode='PX')

        rgb = myCam.imageBufferRGB
        depth = myCam.imageBufferDepthPX
        
        # 2. 优化：只处理图像中央区域，提高处理速度
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        # 针对红色设置掩码
        mask = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
        
        obs_detected = False
        dist = 999
        obs_center = None
        obs_bbox = None

        if np.any(mask):
            M = cv2.moments(mask)
            if M["m00"] > 500:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                obs_center = (cX, cY)
                
                # 测距
                depth_cX, depth_cY = rgb_to_depth_pixel(cX, cY, rgb.shape, depth.shape)
                val = depth[depth_cY, depth_cX][0] if isinstance(
                    depth[depth_cY, depth_cX], np.ndarray) else depth[depth_cY, depth_cX]
                dist = (val * 39.0 / 255.0) / 10.0
                
                # 逻辑：只要在视野中心且距离小于阈值，立即触发
                if dist < LOOK_AHEAD_DIST and 200 < cX < 400:
                    obs_detected = True

                ys, xs = np.where(mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    obs_bbox = (x_min, y_min, x_max, y_max)

        # 3. 避障状态机
        curr_t = get_time()
        throttle = BASE_SPEED
        steering = 0.0

        if state == 'FORWARD':
            if obs_detected:
                avoid_dir = -1 if current_lane == 'LEFT' else 1
                turn_text = '右转' if avoid_dir == -1 else '左转'
                lane_text = '左车道' if current_lane == 'LEFT' else '右车道'
                print(f"!!! 发现障碍: {dist:.2f}m - 当前{lane_text}，立即{turn_text}避障")
                state = 'STEP1'
                state_timer = curr_t
        
        elif state == 'STEP1': # 斜出
            steering = avoid_dir * AVOID_STEER
            if curr_t - state_timer > T_OUT: 
                state = 'STEP2'; 
                state_timer = curr_t
        
        elif state == 'STEP2': # 回正平行
            steering = -avoid_dir * AVOID_STEER
            if curr_t - state_timer > T_FIX: 
                state = 'STEP3'; 
                state_timer = curr_t
        
        elif state == 'STEP3': # 直行通过
            steering = 0.0
            if curr_t - state_timer > T_PASS:
                current_lane = 'RIGHT' if current_lane == 'LEFT' else 'LEFT'
                state = 'FORWARD'; 

        myCar.read_write_std(throttle, steering)

        # 4. 显示反馈
        display_img = rgb.copy()
        lane_text = 'LEFT' if current_lane == 'LEFT' else 'RIGHT'
        cv2.putText(display_img, f"State: {state} Lane: {lane_text} Dist: {dist:.1f}m", (20, 40), 1, 1.5, (0,255,0), 2)

        if obs_bbox is not None and obs_center is not None:
            x_min, y_min, x_max, y_max = obs_bbox
            warning_color = get_warning_color(dist)

            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), warning_color, 3)
            cv2.circle(display_img, obs_center, 5, warning_color, -1)

            warning_text = f"{dist:.2f}m"
            if dist < 1.0:
                status_text = "DANGER"
            elif dist < 1.5:
                status_text = "WARNING"
            else:
                status_text = "SAFE"

            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_bg_top = max(y_min - 30, 0)
            status_y = max(y_min - 35, 15)
            cv2.rectangle(display_img, (x_min, text_bg_top), (x_min + text_size[0] + 10, y_min), warning_color, -1)
            cv2.putText(display_img, warning_text, (x_min + 5, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, status_text, (x_min, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warning_color, 2)

        cv2.imshow('RGB', display_img)
        cv2.waitKey(1)

finally:
    myCar.read_write_std(0, 0)
    myCar.terminate()
    myCam.terminate()