% 旋转倒立摆控制实验 - 最优控制(LQR)算法
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
 
C = [1 0 0 0;
     0 1 0 0];
     
D = [0; 0];

% ==== 2. 判断系统可控性 ====
Pc = ctrb(A, B);
if rank(Pc) == length(A)
    disp('系统状态可控，可以进行LQR最优控制。');
else
    disp('系统状态不可控，无法进行极点配置！');
    return;
end

% ==== 3. 配置性能指标加权矩阵 Q 和 R ====
Q = diag([20 200 1 50]);
R = 1; 
disp('Q 矩阵:');
disp(Q);
disp(['R 矩阵: ', num2str(R)]);

% ==== 4. 计算LQR状态反馈增益矩阵 K ====
[K, S, E] = lqr(A, B, Q, R);
disp('采用LQR算法求得的状态反馈增益矩阵 K 为：');
disp(K);

% 输出此时系统的闭环极点 E，相当于间接得到的配置结果
disp('LQR控制对应的闭环极点(E)为：');
disp(E);

% ==== 5. 画出LQR控制前后的初始状态响应仿真曲线 ====
sys_open = ss(A, B, C, D);           % 开环系统
A_closed = A - B * K;                % LQR状态反馈网络
sys_closed = ss(A_closed, B, C, D);  % 闭环系统

t = 0:0.01:5;
x0 = [0; 0.1; 0; 0]; % 设置系统初始偏移，模拟小范围角度误差 (0.1rad)

[y_open, t_open] = initial(sys_open, x0, t);
[y_closed, t_closed] = initial(sys_closed, x0, t);

% 创建图形窗口
figure('Name', 'LQR最优控制响应比较', 'Position', [100, 100, 1000, 500]);

subplot(2, 2, 1);
plot(t_open, y_open(:, 1));
title('LQR控制前');
ylabel('\theta (rad)');
grid on;

subplot(2, 2, 3);
plot(t_open, y_open(:, 2));
xlabel('时间(s)');
ylabel('\alpha (rad)');
grid on;

subplot(2, 2, 2);
plot(t_closed, y_closed(:, 1));
title('LQR控制后');
ylabel('\theta (rad)');
grid on;

subplot(2, 2, 4);
plot(t_closed, y_closed(:, 2));
xlabel('时间(s)');
ylabel('\alpha (rad)');
grid on;

% 可选：为主图添加一个全局标题（使用 sgtitle）
sgtitle('旋转倒立摆最优控制响应比较 (初始状态x_0 = [0, 0.1, 0, 0])');