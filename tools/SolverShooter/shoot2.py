import numpy as np
import matplotlib.pyplot as plt
import math

# ================= 1. 物理参数配置 (由C++逻辑移植) =================
V0 = 22.3                 # 枪口初速 (m/s)
TYPE = 1                  # 弹丸类型: 0为42mm(英雄), 1为步兵(17mm)

MIN_PITCH = -math.pi / 6  # 最小pitch限位 (rad)
MAX_PITCH = math.pi / 4   # 最大pitch限位 (rad)
MIN_Y = -1.0              # 最小高度 (m)

MAX_ERROR = 0.005         # 允许误差 (m)
ERROR_LEVEL = 5           # 误差等级
GUN = 0.0                # 枪口到pitch轴电机的距离 (m)

G = 9.8                   # 重力加速度 (m/s^2)
STEP = 0.0001             # RK4步长 (s)

# 根据弹丸类型推导阻力系数与质量，为了兼容旧版代码输出，单独赋值M
if TYPE == 0:
    # 英雄 (42mm弹丸)
    K = 1.205 * 0.40 * 0.0425 * 0.0425 / (2 * 0.0445)
    M = 0.0445 
else:
    # 步兵 (17mm弹丸)
    K = 1.205 * 0.47 * 0.0168 * 0.0168 / (2 * 0.0032)
    M = 0.0032

# ================= 2. 核心物理计算函数 (基于C++ RK4仿真移植) =================

def rk4_step_inline(sx, sy, svx, svy, h):
    """内联优化的RK4单步积分，避免Python对象频繁创建带来的严重开销"""
    # k1
    v1 = math.sqrt(svx**2 + svy**2)
    k1_x, k1_y = svx, svy
    k1_vx, k1_vy = -K * v1 * svx, -G - K * v1 * svy
    
    # k2
    s2_vx, s2_vy = svx + 0.5 * h * k1_vx, svy + 0.5 * h * k1_vy
    v2 = math.sqrt(s2_vx**2 + s2_vy**2)
    k2_x, k2_y = s2_vx, s2_vy
    k2_vx, k2_vy = -K * v2 * s2_vx, -G - K * v2 * s2_vy
    
    # k3
    s3_vx, s3_vy = svx + 0.5 * h * k2_vx, svy + 0.5 * h * k2_vy
    v3 = math.sqrt(s3_vx**2 + s3_vy**2)
    k3_x, k3_y = s3_vx, s3_vy
    k3_vx, k3_vy = -K * v3 * s3_vx, -G - K * v3 * s3_vy
    
    # k4
    s4_vx, s4_vy = svx + h * k3_vx, svy + h * k3_vy
    v4 = math.sqrt(s4_vx**2 + s4_vy**2)
    k4_x, k4_y = s4_vx, s4_vy
    k4_vx, k4_vy = -K * v4 * s4_vx, -G - K * v4 * s4_vy
    
    # 更新状态
    nx = sx + h * (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6.0
    ny = sy + h * (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6.0
    nvx = svx + h * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6.0
    nvy = svy + h * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6.0
    
    return nx, ny, nvx, nvy

def get_real_pitch(target_x, target_y, v0):
    """(替换为C++逻辑) 计算有空气阻力下的真实 Pitch 角 (单位：弧度) - 级联容差二分搜索"""
    for i in range(1, ERROR_LEVEL + 1):
        error = MAX_ERROR / ERROR_LEVEL * i
        
        pitch_top = MAX_PITCH
        pitch_low = MIN_PITCH
        
        ans_pitch = None
        
        while (pitch_top - pitch_low) > 0.001:
            pitch_binary = (pitch_top + pitch_low) / 2
            x_b = -GUN * math.cos(pitch_binary)
            y_b = -GUN * math.sin(pitch_binary)
            x_to_gun = target_x + x_b
            y_to_gun = target_y + y_b
            
            sx, sy = 0.0, 0.0
            svx = v0 * math.cos(pitch_binary)
            svy = v0 * math.sin(pitch_binary)
            
            hit = False
            while sy >= MIN_Y - 1:
                sx, sy, svx, svy = rk4_step_inline(sx, sy, svx, svy, STEP)
                
                # 检查是否击中目标容差圈
                if (sx - x_to_gun)**2 + (sy - y_to_gun)**2 <= error**2:
                    ans_pitch = pitch_binary
                    hit = True
                    break
                
                # 飞过目标X轴
                if sx >= x_to_gun:
                    if sy > y_to_gun:
                        pitch_top = pitch_binary
                    else:
                        pitch_low = pitch_binary
                    break
                # 提前坠地且没达到X轴
                elif sy < MIN_Y - 1 and sx < x_to_gun:
                    pitch_low = pitch_binary
                    break
            
            if hit:
                break
                
        if ans_pitch is not None:
            return ans_pitch
            
    return None

def get_ideal_pitch(d, h, v):
    """计算无空气阻力下的理想抛物线 Pitch 角 (单位：弧度)，作为基准被减数"""
    a = (G * d**2) / (2 * v**2)
    b = -d
    c = h + a
    delta = b**2 - 4*a*c
    if delta < 0: return None
    u = (-b - np.sqrt(delta)) / (2*a)
    return np.arctan(u)

# ================= 3. 生成数据集并拟合 (保持不变) =================

d_values = np.linspace(1.0, 8.0, 50)
h_values = np.linspace(-1.5, 1.5, 10)

data_d = []
data_delta_theta_rad = []  # 存的是弧度差

print(f"正在使用 RK4 动力学引擎生成 {len(d_values) * len(h_values)} 组数据（因为步长较小，这可能需要几十秒，请稍候）...")
for h in h_values:
    for d in d_values:
        ideal_p = get_ideal_pitch(d, h, V0)
        real_p = get_real_pitch(d, h, V0)
        
        if ideal_p is not None and real_p is not None:
            data_d.append(d)
            data_delta_theta_rad.append(real_p - ideal_p)

# 二次多项式拟合
coeffs = np.polyfit(data_d, data_delta_theta_rad, 2)
A_coef, B_coef, C_coef = coeffs

print("\n================= 1. 拟合结果 =================\n")
print(f"当前弹速: {V0} m/s | 弹重: {M*1000} g")
print(f"非线性角度补偿公式 (单位: 弧度 Radian):")
print(f"Delta_Pitch = ({A_coef:.6f}) * d^2 + ({B_coef:.6f}) * d + ({C_coef:.6f})\n")
print("C语言/C++ 宏替换代码:")
print(f"#define COMPENSATE_A  {A_coef:.6f}f")
print(f"#define COMPENSATE_B  {B_coef:.6f}f")
print(f"#define COMPENSATE_C  {C_coef:.6f}f")
print(f"a = {A_coef:.8f}, b = {B_coef:.8f}, c = {C_coef:.8f};")

# ================= 4. 误差分析 (Error Analysis) =================
print("\n================= 2. 精度评估 (最大物理落差) =================")
print("对比拟合曲线与真实空气动力学模型(RK4)的最大偏差：\n")

test_distances = [3.0, 5.0, 7.0, 8.0]
tolerance_d = 0.1 # 距离查找容差

for target_d in test_distances:
    max_err_rad = 0.0
    
    for i in range(len(data_d)):
        if abs(data_d[i] - target_d) < tolerance_d:
            fit_val = np.polyval(coeffs, data_d[i])
            err_rad = abs(data_delta_theta_rad[i] - fit_val)
            if err_rad > max_err_rad:
                max_err_rad = err_rad
                
    # 将弧度误差转化为靶面物理误差 (厘米) -> d * tan(err_rad) * 100
    max_err_cm = target_d * np.tan(max_err_rad) * 100
    
    # 打印时顺便把弧度转回角度，方便人类阅读直观感受
    max_err_deg = np.degrees(max_err_rad)
    print(f"距离 {target_d}m 处 -> 最大角度误差: {max_err_deg:.4f}° ({max_err_rad:.6f} rad) | 最大靶面偏差: {max_err_cm:.2f} cm")

print("\n==============================================================")

# ================= 5. 绘图可视化 =================
plt.figure(figsize=(10, 6))
plt.scatter(data_d, data_delta_theta_rad, color='gray', alpha=0.5, label='Theoretical Residuals (All Heights)')
d_line = np.linspace(1, 8, 100)
delta_line = np.polyval(coeffs, d_line)
plt.plot(d_line, delta_line, color='red', linewidth=3, label=f'Fit: {A_coef:.5f}d² + {B_coef:.5f}d + {C_coef:.5f}')
plt.title(f'RK4 Air Resistance Pitch Compensation (V0 = {V0}m/s)')
plt.xlabel('Horizontal Distance d (m)')
plt.ylabel('Delta Pitch Angle (Radians)')
plt.grid(True)
plt.legend()
plt.show()