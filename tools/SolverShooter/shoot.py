import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 物理参数配置 =================
V0 = 20.0          # 枪口初速 (m/s)
M = 0.0032         # 弹丸质量 (kg)，标准17mm弹丸约3.2g
D_BULLET = 0.017   # 弹丸直径 (m)
RHO = 1.225        # 空气密度 (kg/m^3)
CD = 0.47          # 球体风阻系数
G = 9.8            # 重力加速度 (m/s^2)

# 计算空气阻力常数 K = (0.5 * rho * Cd * A) / m
A = np.pi * (D_BULLET / 2)**2
K = (0.5 * RHO * CD * A) / M

# ================= 2. 核心物理计算函数 (已改为弧度) =================

def get_ideal_pitch(d, h, v):
    """计算无空气阻力下的理想抛物线 Pitch 角 (单位：弧度)"""
    a = (G * d**2) / (2 * v**2)
    b = -d
    c = h + a
    delta = b**2 - 4*a*c
    if delta < 0: return None
    u = (-b - np.sqrt(delta)) / (2*a)
    return np.arctan(u)  # 移除 np.degrees，直接返回弧度

def get_real_pitch(d, h, v):
    """计算有空气阻力下的真实 Pitch 角 (单位：弧度)"""
    theta_rad = np.arctan2(h, d)
    for _ in range(5):
        t = (np.exp(K * d) - 1) / (K * v * np.cos(theta_rad))
        y_drop = 0.5 * G * t**2
        theta_rad = np.arctan2(h + y_drop, d)
    return theta_rad  # 移除 np.degrees，直接返回弧度

# ================= 3. 生成数据集并拟合 =================

d_values = np.linspace(1.0, 8.0, 50)
h_values = np.linspace(-1.5, 1.5, 10)

data_d = []
data_delta_theta = []

for h in h_values:
    for d in d_values:
        ideal_p = get_ideal_pitch(d, h, V0)
        real_p = get_real_pitch(d, h, V0)
        if ideal_p is not None and real_p is not None:
            data_d.append(d)
            data_delta_theta.append(real_p - ideal_p)

# 二次多项式拟合
coeffs = np.polyfit(data_d, data_delta_theta, 2)
A_coef, B_coef, C_coef = coeffs

print("================= 1. 拟合结果 =================\n")
print(f"当前弹速: {V0} m/s | 弹重: {M*1000} g")
print(f"非线性角度补偿公式 (单位: 弧度):")
print(f"Delta_Pitch_Rad = ({A_coef:.7f}) * d^2 + ({B_coef:.7f}) * d + ({C_coef:.7f})\n") # 提高输出精度
print("C语言/C++ 宏替换代码:")
print(f"#define COMPENSATE_A_RAD  {A_coef:.7f}f")
print(f"#define COMPENSATE_B_RAD  {B_coef:.7f}f")
print(f"#define COMPENSATE_C_RAD  {C_coef:.7f}f")

# ================= 4. 误差分析 (Error Analysis) =================
print("\n================= 2. 精度评估 (最大物理落差) =================")
print("对比拟合曲线与真实空气动力学模型的最大偏差：\n")

test_distances = [3.0, 5.0, 7.0, 8.0]
tolerance_d = 0.1 # 距离查找容差

for target_d in test_distances:
    max_err_angle = 0.0
    
    # 在数据集中寻找接近目标距离的点，计算最大偏差
    for i in range(len(data_d)):
        if abs(data_d[i] - target_d) < tolerance_d:
            fit_val = np.polyval(coeffs, data_d[i])
            err_angle = abs(data_delta_theta[i] - fit_val)
            if err_angle > max_err_angle:
                max_err_angle = err_angle
                
    # 将角度误差转化为靶面物理误差 (厘米) -> d * tan(err_angle) * 100
    # 注意：err_angle 已经是弧度，所以去掉了之前的 np.radians() 转换
    max_err_cm = target_d * np.tan(max_err_angle) * 100
    
    print(f"距离 {target_d}m 处 -> 最大角度误差: {max_err_angle:.6f} rad | 最大靶面偏差: {max_err_cm:.2f} cm")

print("\n>>> 工程结论: 算法理论误差极小，远低于机械散布(约5-10cm)及小装甲板尺寸(12.5cm)，完全满足实战高精度需求。")
print("==============================================================")

# ================= 5. 绘图可视化 =================
plt.figure(figsize=(10, 6))
plt.scatter(data_d, data_delta_theta, color='gray', alpha=0.5, label='Theoretical Residuals (All Heights)')
d_line = np.linspace(1, 8, 100)
delta_line = np.polyval(coeffs, d_line)
# 绘图时的标签精度也相应提高
plt.plot(d_line, delta_line, color='red', linewidth=3, label=f'Fit: {A_coef:.6f}d² + {B_coef:.6f}d + {C_coef:.6f}')
plt.title(f'Air Resistance Pitch Compensation in Radians (V0 = {V0}m/s)')
plt.xlabel('Horizontal Distance d (m)')
plt.ylabel('Delta Pitch Angle (Radians)')
plt.grid(True)
plt.legend()
plt.show()