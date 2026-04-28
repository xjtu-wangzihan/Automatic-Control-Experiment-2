from pal.products.qcar import QCar, QCarRealSense
import time
import numpy as np
import cv2
import signal

# 这个脚本实现了两个任务的结合：
# 1. 正常情况下，车辆根据车道中心的彩色引导线进行循迹。
# 2. 当前车道前方发现红色障碍物后，车辆执行一个固定时序的换道避障动作。
# 3. 换道完成后，切换到另一条车道对应的引导线，继续循迹前进。

# --- 车辆基本运动参数 ---
# 基础前进速度。值越大，车速越快，但循迹和避障对参数误差会更敏感。
BASE_SPEED = 0.15
# 避障阶段的固定转向量。STEP1/STEP2 期间直接使用这个值，不走循迹控制。
AVOID_STEER = 0.15
# 前方障碍触发距离阈值，单位米。小于该值且位于画面中央时触发避障。
LOOK_AHEAD_DIST = 1.4

# --- 引导线巡迹参数 ---
# 比例项：车偏离引导线越多，纠偏转向越强。
GUIDE_KP = 0.7
# 微分项：抑制转向变化过快，减少左右来回摆动。
GUIDE_KD = 0.10
# 循迹输出转向限幅，避免单次修正过猛。
GUIDE_STEER_LIMIT = 0.22
# 只在图像下半部分搜索引导线，减少远处噪声和无关区域干扰。
GUIDE_ROI_Y_START = 0.45
# 单行里至少要有这么多像素才认为这一行真的检测到了线。
GUIDE_MIN_ROW_PIXELS = 5
# 至少需要这么多采样行，才认为本帧的引导线中心可靠。
GUIDE_MIN_SAMPLES = 6
# 引导线中心平滑系数。越大越稳，但响应越慢。
GUIDE_SMOOTH_ALPHA = 0.72
# 短暂丢线时，允许沿用上一次结果的最大帧数。
GUIDE_HOLD_FRAMES = 8
# 换道后进入循迹过渡期的持续时间，防止刚切换目标线时打方向过猛。
GUIDE_SWITCH_SETTLE_TIME = 0.60
# 过渡期内对转向再做一次更严格的限幅。
GUIDE_SWITCH_STEER_LIMIT = 0.12

# 左车道中心引导线使用黄色，右车道中心引导线使用绿色。
# 这里的阈值用于在 HSV 空间中分割这两条引导线。
YELLOW_LOWER = np.array([18, 120, 120])
YELLOW_UPPER = np.array([40, 255, 255])
GREEN_LOWER = np.array([40, 80, 80])
GREEN_UPPER = np.array([90, 255, 255])

# 红色障碍物的阈值。由于 HSV 的红色跨越 0/180 两端，因此要分成两段检测后再合并。
RED_LOWER_1 = np.array([0, 100, 100])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([160, 100, 100])
RED_UPPER_2 = np.array([180, 255, 255])

# 仅用于显示调试信息时的颜色映射，不参与控制。
LANE_COLORS_BGR = {
    'LEFT': (0, 255, 255),
    'RIGHT': (0, 255, 0),
}
# 仅用于打印中文提示。
LANE_NAMES_CN = {
    'LEFT': '左车道',
    'RIGHT': '右车道',
}

# --- 避障状态机时间 (秒) ---
# STEP1：从当前车道斜向切出去。
T_OUT = 0.849
# STEP2：反向打一把方向，使车身与目标车道大致平行。
T_FIX = 0.848
# STEP3：切换到新车道的循迹控制后，再给一个很短的通过时间。
T_PASS = 0.05

# --- 硬件初始化 ---
myCar = QCar()
print("正在连接摄像头...")
myCam = QCarRealSense(mode='RGB, Depth')

# RealSense 启动后的前几秒数据通常不够稳定，这里预热一下。
print("等待 RealSense 预热...")
time.sleep(3)

# 全局运行状态。
startTime = time.time()
state = 'FORWARD'
state_timer = 0
stop_requested = False
avoid_dir = -1
# 场景初始化时车辆出生在左车道，所以初始跟踪目标也是左车道引导线。
current_lane = 'LEFT'


def handle_sigint(*args):
    """响应 Ctrl+C，通知主循环安全退出。"""
    global stop_requested
    stop_requested = True
    print("检测到 Ctrl+C，安全停止实验...")

signal.signal(signal.SIGINT, handle_sigint)


def get_time():
    """返回程序已经运行的秒数。"""
    return time.time() - startTime


def get_warning_color(distance):
    """根据障碍物距离返回显示颜色。"""
    if distance < 1.0:
        return (0, 0, 255)
    if distance < 1.5:
        return (0, 255, 255)
    return (0, 255, 0)


def smooth_center(prev_value, new_value, alpha):
    """对检测到的引导线中心做一阶低通平滑。

    作用：
    1. 降低单帧识别抖动带来的方向来回抖动。
    2. 让转向更连续，避免控制量忽大忽小。

    参数：
    - prev_value: 上一帧平滑后的中心位置
    - new_value: 当前帧新检测到的中心位置
    - alpha: 平滑系数，越接近 1 越依赖上一帧

    返回：
    - 平滑后的中心位置
    """
    # 当前帧没检测到时，沿用上一帧结果。
    if new_value is None:
        return prev_value
    # 第一次得到有效值时，直接使用，不做平滑。
    if prev_value is None:
        return float(new_value)
    # 标准指数滑动平均。
    return alpha * float(prev_value) + (1.0 - alpha) * float(new_value)


def estimate_single_guide_center(roi_mask, roi_y0):
    """从单条引导线的二值图中估计该线的横向中心。

    输入是某种颜色引导线在 ROI 中的二值掩码。
    做法是从下往上逐行采样，每一行求出该行线条像素的平均 x，
    然后对所有采样行做加权平均，得到这一帧的引导线中心位置。

    之所以给靠近车辆的行更高权重，是因为：
    1. 近处像素对当前转向更敏感
    2. 远处透视变形更大，误差也更大
    """
    sample_xs = []
    sample_weights = []
    sample_points = []
    roi_h = roi_mask.shape[0]

    # 从图像底部往上每隔 4 行采一次样。
    for row in range(roi_h - 1, -1, -4):
        xs = np.where(roi_mask[row] > 0)[0]
        # 本行像素太少时，通常是噪声或断裂，不参与中心估计。
        if xs.size < GUIDE_MIN_ROW_PIXELS:
            continue

        x_center = float(np.mean(xs))
        y_abs = int(row + roi_y0)
        sample_xs.append(x_center)
        # 越靠近图像底部，权重越高。
        sample_weights.append(1.0 + row / max(roi_h - 1, 1))
        sample_points.append((int(round(x_center)), y_abs))

    # 有效采样不足时，认为这一帧没有稳定检测到该引导线。
    if len(sample_xs) < GUIDE_MIN_SAMPLES:
        return None, sample_points

    guide_center = int(round(np.average(sample_xs, weights=sample_weights)))
    return guide_center, sample_points


def detect_guide_lines(rgb_img):
    """检测左右两条车道中心引导线。

    返回：
    - guide_centers: 左/右引导线的横向中心
    - guide_points: 每条线逐行采样得到的点，供可视化调试
    - guide_masks: 左/右引导线完整二值图，供显示 GuideMask
    - roi_y0: ROI 起始行，便于把检测结果画回原图
    """
    # OpenCV 默认读取的是 BGR，这里转 HSV 便于做颜色分割。
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    img_h, img_w = rgb_img.shape[:2]
    roi_y0 = int(img_h * GUIDE_ROI_Y_START)
    # 只保留图像下半部分，提高鲁棒性和运行速度。
    roi_hsv = hsv_img[roi_y0:, :]

    yellow_roi = cv2.inRange(roi_hsv, YELLOW_LOWER, YELLOW_UPPER)
    green_roi = cv2.inRange(roi_hsv, GREEN_LOWER, GREEN_UPPER)

    kernel = np.ones((3, 3), np.uint8)
    # 先开运算去掉零碎噪点，再闭运算连通小断裂。
    yellow_roi = cv2.morphologyEx(yellow_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    yellow_roi = cv2.morphologyEx(yellow_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
    green_roi = cv2.morphologyEx(green_roi, cv2.MORPH_OPEN, kernel, iterations=1)
    green_roi = cv2.morphologyEx(green_roi, cv2.MORPH_CLOSE, kernel, iterations=2)

    left_center, left_points = estimate_single_guide_center(yellow_roi, roi_y0)
    right_center, right_points = estimate_single_guide_center(green_roi, roi_y0)

    yellow_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    green_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    yellow_mask[roi_y0:, :] = yellow_roi
    green_mask[roi_y0:, :] = green_roi

    guide_centers = {
        'LEFT': left_center,
        'RIGHT': right_center,
    }
    guide_points = {
        'LEFT': left_points,
        'RIGHT': right_points,
    }
    guide_masks = {
        'LEFT': yellow_mask,
        'RIGHT': green_mask,
    }
    return guide_centers, guide_points, guide_masks, roi_y0


try:
    # 平滑后的左右引导线中心，允许在帧与帧之间连续过渡。
    filtered_guide_centers = {'LEFT': None, 'RIGHT': None}
    # 记录每条线最后一次被成功检测到的帧号，用于短暂丢线时保持控制连续性。
    guide_last_seen_frame = {'LEFT': -9999, 'RIGHT': -9999}
    # 上一帧的归一化误差，用来计算微分项。
    prev_guide_error_norm = 0.0
    # 换道后过渡限幅结束的时刻。
    lane_switch_settle_until = 0.0
    frame_index = 0

    while get_time() < 60 and not stop_requested:
        frame_index += 1

        # 同时读取 RGB 和深度。RGB 用于识别引导线和障碍物，深度用于估计障碍距离。
        myCam.read_RGB()
        myCam.read_depth(dataMode='PX')

        rgb = myCam.imageBufferRGB
        depth = myCam.imageBufferDepthPX
        image_center_x = rgb.shape[1] // 2

        # 提取左右引导线的中心位置和调试信息。
        guide_centers_raw, guide_points, guide_masks, guide_roi_y0 = detect_guide_lines(rgb)

        # 对每条线的中心做平滑，减少单帧抖动。
        for lane_name in ('LEFT', 'RIGHT'):
            raw_center = guide_centers_raw[lane_name]
            if raw_center is not None:
                filtered_guide_centers[lane_name] = smooth_center(
                    filtered_guide_centers[lane_name],
                    raw_center,
                    GUIDE_SMOOTH_ALPHA,
                )
                guide_last_seen_frame[lane_name] = frame_index

        # 当前应该跟踪哪条线，完全由 current_lane 决定。
        target_lane = current_lane
        target_guide_x = None
        guide_status = 'LOST'

        # 优先使用当前帧刚检测到的线。
        if guide_centers_raw[target_lane] is not None:
            target_guide_x = int(round(filtered_guide_centers[target_lane]))
            guide_status = 'TRACK'
        # 如果当前帧丢线，但距离上次检测成功还不久，则短时间继续沿用旧结果。
        elif (
            filtered_guide_centers[target_lane] is not None
            and frame_index - guide_last_seen_frame[target_lane] <= GUIDE_HOLD_FRAMES
        ):
            target_guide_x = int(round(filtered_guide_centers[target_lane]))
            guide_status = 'HOLD'
        else:
            # 长时间丢线后清空历史值，避免一直拿旧线误导控制器。
            filtered_guide_centers[target_lane] = None
            prev_guide_error_norm = 0.0

        guide_error_px = 0
        guide_steer = 0.0
        if target_guide_x is not None:
            # 误差定义为“目标线相对画面中心的偏移”。
            guide_error_px = target_guide_x - image_center_x
            # 归一化到 [-1, 1] 左右，方便参数更有可迁移性。
            guide_error_norm = guide_error_px / (rgb.shape[1] / 2.0)
            guide_error_diff = guide_error_norm - prev_guide_error_norm
            # PD 控制：P 负责纠偏，D 负责抑制过冲和摆动。
            guide_steer = float(
                np.clip(
                    -(GUIDE_KP * guide_error_norm + GUIDE_KD * guide_error_diff),
                    -GUIDE_STEER_LIMIT,
                    GUIDE_STEER_LIMIT,
                )
            )
            prev_guide_error_norm = guide_error_norm

        # 障碍物检测单独基于红色阈值完成，与引导线检测互不影响。
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1),
            cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2),
        )

        obs_detected = False
        dist = 999.0
        obs_center = None
        obs_bbox = None

        if np.any(red_mask):
            M = cv2.moments(red_mask)
            if M["m00"] > 500:
                # 用红色区域质心作为障碍物代表点。
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                obs_center = (cX, cY)

                # 从深度图取对应像素，转换为近似米制距离。
                depth_value = depth[cY, cX][0] if isinstance(depth[cY, cX], np.ndarray) else depth[cY, cX]
                dist = (depth_value * 39.0 / 255.0) / 10.0

                # 只在障碍足够近且位于车辆正前方视野时才触发避障。
                if dist < LOOK_AHEAD_DIST and 200 < cX < 400:
                    obs_detected = True

                # 提取整块红色区域的包围盒，用于显示。
                ys, xs = np.where(red_mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    obs_bbox = (x_min, y_min, x_max, y_max)

        # 默认情况下，车辆按循迹控制输出方向。
        curr_t = get_time()
        throttle = BASE_SPEED
        steering = guide_steer

        if state == 'FORWARD':
            # 正常直行阶段：持续监视前方是否出现障碍。
            if obs_detected:
                # 左车道遇障碍就向右切，右车道遇障碍就向左切。
                avoid_dir = -1 if current_lane == 'LEFT' else 1
                turn_text = '右转' if avoid_dir == -1 else '左转'
                print(
                    f"!!! 发现障碍: {dist:.2f}m - 当前{LANE_NAMES_CN[current_lane]}，立即{turn_text}避障"
                )
                state = 'STEP1'
                state_timer = curr_t

        elif state == 'STEP1':
            # 第一步：向目标车道方向斜切出去。
            steering = avoid_dir * AVOID_STEER
            if curr_t - state_timer > T_OUT:
                state = 'STEP2'
                state_timer = curr_t

        elif state == 'STEP2':
            # 第二步：反打一把，让车身大致与新车道平行。
            steering = -avoid_dir * AVOID_STEER
            if curr_t - state_timer > T_FIX:
                # 在这里正式切换当前车道，从下一阶段开始跟踪另一条引导线。
                current_lane = 'RIGHT' if current_lane == 'LEFT' else 'LEFT'
                prev_guide_error_norm = 0.0
                lane_switch_settle_until = curr_t + GUIDE_SWITCH_SETTLE_TIME
                state = 'STEP3'
                state_timer = curr_t

        elif state == 'STEP3':
            # 第三步：进入新车道后，立刻由循迹控制器接管方向。
            steering = guide_steer
            if curr_t - state_timer > T_PASS:
                state = 'FORWARD'

        # 刚换道完成时，就算循迹误差还比较大，也先限制一下方向，避免猛打方向。
        if state in ('FORWARD', 'STEP3') and curr_t < lane_switch_settle_until:
            steering = float(np.clip(steering, -GUIDE_SWITCH_STEER_LIMIT, GUIDE_SWITCH_STEER_LIMIT))

        # 向车辆写入纵向和横向控制。
        myCar.read_write_std(throttle, steering)

        # -----------------------------
        # 以下部分全部是可视化与调试显示
        # -----------------------------
        display_img = rgb.copy()
        mask_vis = np.zeros_like(display_img)
        mask_vis[guide_masks['LEFT'] > 0] = LANE_COLORS_BGR['LEFT']
        mask_vis[guide_masks['RIGHT'] > 0] = LANE_COLORS_BGR['RIGHT']

        # 画出图像中心参考线，便于观察误差方向和大小。
        cv2.line(
            display_img,
            (image_center_x, guide_roi_y0),
            (image_center_x, display_img.shape[0] - 1),
            (255, 0, 0),
            2,
        )

        for lane_name in ('LEFT', 'RIGHT'):
            # 逐行采样点显示为小圆点，用来看颜色分割是否稳定。
            for x_pt, y_pt in guide_points[lane_name]:
                cv2.circle(display_img, (x_pt, y_pt), 2, LANE_COLORS_BGR[lane_name], -1)

            filtered_center = filtered_guide_centers[lane_name]
            if filtered_center is not None:
                center_x = int(round(filtered_center))
                # 当前目标车道画得更粗，便于区分控制器实际在跟哪条线。
                line_width = 4 if lane_name == target_lane else 2
                cv2.line(
                    display_img,
                    (center_x, guide_roi_y0),
                    (center_x, display_img.shape[0] - 1),
                    LANE_COLORS_BGR[lane_name],
                    line_width,
                )

        if target_guide_x is not None:
            # 白线表示当前真正用于控制的目标中心。
            cv2.line(
                display_img,
                (target_guide_x, guide_roi_y0),
                (target_guide_x, display_img.shape[0] - 1),
                (255, 255, 255),
                2,
            )

        cv2.putText(
            display_img,
            f"State: {state} Lane: {current_lane} Dist: {dist:.1f}m",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_img,
            f"Guide: {guide_status} Target: {target_lane} Err: {guide_error_px:+d}px Steer: {guide_steer:+.3f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        if obs_bbox is not None and obs_center is not None:
            x_min, y_min, x_max, y_max = obs_bbox
            warning_color = get_warning_color(dist)

            # 障碍物显示框和质心点。
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
            cv2.rectangle(
                display_img,
                (x_min, text_bg_top),
                (x_min + text_size[0] + 10, y_min),
                warning_color,
                -1,
            )
            cv2.putText(
                display_img,
                warning_text,
                (x_min + 5, y_min - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display_img,
                status_text,
                (x_min, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                warning_color,
                2,
            )

        cv2.imshow('RGB', display_img)
        cv2.imshow('GuideMask', mask_vis)
        cv2.waitKey(1)

finally:
    # 无论正常结束还是中途异常，都先停车，再释放资源。
    myCar.read_write_std(0, 0)
    myCar.terminate()
    myCam.terminate()
