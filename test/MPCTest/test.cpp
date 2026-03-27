#include "../../rm-main/planner/planner.hpp"
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std::chrono_literals;

void run_simulation() {
    // 1. 初始化 Planner (记得准备好对应的 yaml 配置文件)
    MPC::Planner planner("../config/planner.yaml");

    // 2. 伪造一个初始的机器人状态
    // 假设目标距离我们 3 米，正在以 w = 2.0 rad/s 的角速度自转，并且以 1.0 m/s 的速度向右移动
    Robot fake_robot;
    fake_robot.Speed << 0, 0.0, 0.0, 5.0; // vx, vy, vz, w
    fake_robot.center << 0.0, 3.0, 0.5;     // x, y, z
    // 装甲板初始状态 (theta, radius, h)
    fake_robot.Armors << 0.0, M_PI/2, M_PI, 3*M_PI/2,
                         0.25, 0.20,    0.25,  0.2,
                         0.05, 0.0,    0.05,  0.0;
                         
    auto start_time = std::chrono::steady_clock::now();
    RobotState target(fake_robot, start_time);

    // 3. 打开 CSV 文件准备记录
    std::ofstream file("mpc_simulation_result.csv");
    file << "time,target_yaw,mpc_yaw,target_pitch,mpc_pitch,mpc_yaw_vel,mpc_yaw_acc\n";

    // 4. 开始仿真循环 (模拟 2 秒的运行，步长 10ms)
    double dt = 0.01;
    for (int i = 0; i < 500; ++i) {
        double current_time = i * dt;
        auto now = start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(current_time));
        
        // 更新目标的真实状态 (模拟时间流逝)
        target.Predict(now);

        // 调用你的 MPC 进行规划，假设子弹速度 20m/s
        MPC::Plan plan = planner.plan(target, 20.0);

        // 将结果写入 CSV
        if (plan.control) {
            file << current_time << "," 
                 << plan.target_yaw << "," << plan.yaw << ","
                 << plan.target_pitch << "," << plan.pitch << ","
                 << plan.yaw_vel << "," << plan.yaw_acc << "\n";
        }
    }
    
    file.close();
    std::cout << "Simulation finished. Data saved to mpc_simulation_result.csv" << std::endl;
}

int main() {
    run_simulation();
    return 0;
}