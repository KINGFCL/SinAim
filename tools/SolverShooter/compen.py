import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示 (如果系统中没有此字体可去除或替换为其他可用中文字体)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义距离范围 d (1m 到 8m)，取 500 个点使曲线平滑
d = np.linspace(1, 8, 500)

# 2. 第一组参数 (宏定义宏替换，原 float 型)
A1 =    0.000200
B1 = -0.000137
C1 = 0.000166

# 3. 第二组参数 (const double 型)
A2 = 0.000244
B2 = -0.000182
C2 = 0.000202

# 4. 计算两个函数的值
# 假设 distance_2 代表 distance 的平方
y1 = A1 * (d**2) + B1 * d + C1
y2 = A2 * (d**2) + B2 * d + C2

# 5. 计算误差 (函数1 - 函数2)
error = y1 - y2

# 6. 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 上半图：对比两个函数的值
ax1.plot(d, y1, label='函数 1 (宏定义参数)', color='blue')
ax1.plot(d, y2, label='函数 2 (const double 参数)', color='red', linestyle='--')
ax1.set_title('两组补偿函数的数值对比')
ax1.set_xlabel('距离 Distance (m)')
ax1.set_ylabel('补偿值 Compensation Value')
ax1.legend()
ax1.grid(True)

# 下半图：误差分析
ax2.plot(d, error, label='绝对误差 (函数1 - 函数2)', color='green')
ax2.set_title('误差分析图 (1m - 8m)')
ax2.set_xlabel('距离 Distance (m)')
ax2.set_ylabel('误差 Error (y1 - y2)')
ax2.axhline(0, color='black', linewidth=1, linestyle='--') # 绘制0刻度基准线
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()