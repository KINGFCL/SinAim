import pandas as pd
import matplotlib.pyplot as plt

def plot_results(csv_file):
    # 读取 C++ 输出的数据
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"找不到文件 {csv_file}，请确保 C++ 仿真已成功运行并输出了文件。")
        return

    time = df['time']

    # 创建一个 3 行 1 列的画布
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1. 绘制 Yaw 轴跟踪曲线
    axs[0].plot(time, df['target_yaw'], label='Target Yaw (Predicted)', linestyle='--', color='red')
    axs[0].plot(time, df['mpc_yaw'], label='MPC Output Yaw', color='blue')
    axs[0].set_title('Yaw Tracking Performance')
    axs[0].set_ylabel('Angle (rad)')
    axs[0].legend()
    axs[0].grid(True)

    # 2. 绘制 Pitch 轴跟踪曲线
    axs[1].plot(time, df['target_pitch'], label='Target Pitch (Predicted)', linestyle='--', color='red')
    axs[1].plot(time, df['mpc_pitch'], label='MPC Output Pitch', color='blue')
    axs[1].set_title('Pitch Tracking Performance')
    axs[1].set_ylabel('Angle (rad)')
    axs[1].legend()
    axs[1].grid(True)

    # 3. 绘制控制量输出 (速度与加速度)
    axs[2].plot(time, df['mpc_yaw_vel'], label='MPC Yaw Velocity', color='green')
    axs[2].plot(time, df['mpc_yaw_acc'], label='MPC Yaw Acceleration', color='purple', alpha=0.6)
    axs[2].set_title('Control Outputs')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Vel(rad/s) / Acc(rad/s^2)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results("mpc_simulation_result.csv")