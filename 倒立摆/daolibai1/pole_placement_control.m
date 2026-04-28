% 旋转倒立摆控制实验 - 极点配置算法

clear; clc; close all;

% ==== 1. 倒立摆系统参数矩阵 ====
A = [0 0 1 0;
     0 0 0 1;
     0 149.2751 -0.0104 0;
     0 -261.6091 -0.0103 0];
 
B = [0;
     0;
     49.7275;
     49.1493];
 
C = [1 0 0 0;  % 输出 1: 旋转臂角度 theta
     0 1 0 0]; % 输出 2: 摆杆角度 alpha
     
D = [0; 0];

% ==== 2. 判断系统可控性 ====
Pc = ctrb(A, B);
if rank(Pc) == length(A)
    disp('系统状态可控，可以进行极点配置。');
else
    disp('系统状态不可控，无法进行极点配置！');
    return;
end

% ==== 3. 性能指标转换与期望极点计算 ====
zeta = 0.9;
wn = 6;

% 计算主导极点
p1 = -zeta * wn + 1j * wn * sqrt(1 - zeta^2);
p2 = -zeta * wn - 1j * wn * sqrt(1 - zeta^2);

% 非主导极点选取
p3 = -150;
p4 = -200;

P_desired = [p1, p2, p3, p4];
disp('期望的闭环极点位置为：');
disp(P_desired);

% ==== 4. 极点配置计算反馈增益矩阵 K ====
K = acker(A, B, P_desired);
disp('通过极点配置求得的状态反馈增益矩阵 K 为：');
disp(K);

% ==== 5. 画出极点配置前后的初始状态响应仿真曲线并标注指标 ====
sys_open = ss(A, B, C, D);           
A_closed = A - B * K;                
sys_closed = ss(A_closed, B, C, D);  

t = 0:0.01:5;
x0 = [0; 0.1; 0; 0]; 

[y_open, t_open] = initial(sys_open, x0, t);
[y_closed, t_closed] = initial(sys_closed, x0, t);

% 提取闭环摆杆角度数据 alpha
y_alpha = y_closed(:, 2);
initial_alpha = x0(2); % 初始值 0.1

% --- 计算性能指标 ---
% 1. 超调量 Mp: 响应越过 0 后的最大绝对值
peak_val = min(y_alpha); % 寻找反向峰值
if peak_val < 0
    overshoot_abs = abs(peak_val);
    Mp = (overshoot_abs / abs(initial_alpha)) * 100; 
else
    overshoot_abs = 0;
    Mp = 0;
end

% 2. 调节时间 ts: 响应进入并保持在初始误差 5% 范围内的时刻
threshold = 0.05 * abs(initial_alpha);
idx = find(abs(y_alpha) > threshold, 1, 'last');
if isempty(idx)
    ts = 0;
else
    ts = t_closed(idx);
end

% --- 绘图 ---
figure('Name', '极点配置控制响应与指标标注', 'Position', [100, 100, 1000, 600]);

% 前三个子图保持原样
subplot(2, 2, 1); plot(t_open, y_open(:, 1)); title('极点配置前'); ylabel('\theta (rad)'); grid on;
subplot(2, 2, 3); plot(t_open, y_open(:, 2)); xlabel('时间(s)'); ylabel('\alpha (rad)'); grid on;
subplot(2, 2, 2); plot(t_closed, y_closed(:, 1)); title('极点配置后'); ylabel('\theta (rad)'); grid on;

% 第四个子图：标注 alpha 的指标
subplot(2, 2, 4);
plot(t_closed, y_alpha, 'LineWidth', 1.5);
hold on;
grid on;

% 标注超调点
if overshoot_abs > 0
    [~, p_idx] = min(y_alpha);
    plot(t_closed(p_idx), peak_val, 'ro', 'MarkerSize', 8);
    text(t_closed(p_idx)+0.1, peak_val, ['超调量 M_p = ', num2str(Mp, '%.2f'), '%'], 'Color', 'r');
end

% 标注调节时间
plot(ts, y_alpha(idx), 'bs', 'MarkerSize', 8);
line([ts ts], [0 y_alpha(idx)], 'Color', 'b', 'LineStyle', '--');
text(ts+0.1, 0.02, ['调节时间 t_s = ', num2str(ts, '%.2f'), 's'], 'Color', 'b');

xlabel('时间(s)');
ylabel('\alpha (rad)');

sgtitle('旋转倒立摆极点配置分析');