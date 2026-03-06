#ifndef RERUN_VISUALIZER_HPP
#define RERUN_VISUALIZER_HPP

#include <rerun.hpp>
#include <eigen3/Eigen/Core>
#include <rerun/archetypes/series_lines.hpp>
#include <vector>
#include <string>
#include <cmath>

// 包含你的 Robot 定义
#include "Target.hpp" 

class RerunVisualizer {
private:
    rerun::RecordingStream rec;

public:
    // 构造函数：初始化 Rerun 并唤起 Viewer
    RerunVisualizer(const std::string& app_name = "RoboMaster_AutoAim",
                    const std::string& rerun_path = "./rerun") 
        : rec(app_name) 
    {
        // 配置 Viewer 路径并启动
        rerun::SpawnOptions spawn_opts;
        spawn_opts.executable_name = rerun_path;
        rec.spawn(spawn_opts).exit_on_failure();
    }

    // 析构函数
    ~RerunVisualizer() = default;

    /**
     * @brief 核心更新函数，在 main 的 while 循环中调用
     * @param robot 当前机器人的状态 (包含装甲板、速度、中心点)
     * @param aims  预测出的击打点矩阵 (4x4)
     * @param dt    预测时间
     * @param Gun   当前枪管朝向 (单位向量)
     * @param image 相机原图 (可选，用于同步显示2D画面)
     */
    void update(const Robot& robot, const Eigen::Matrix<double, 4, 4>& aims, double dt, const Eigen::Matrix<double, 3, 1>& Gun) 
    {
        // 1. 记录预测时间 dt
        rec.log("debug/dt", rerun::Scalars(dt));

        // 2. 准备 3D 渲染所需的数据容器
        std::vector<rerun::Position3D> visible_armors;
        std::vector<rerun::Position3D> hidden_armors;
        std::vector<rerun::Position3D> armor_normals;   // 装甲板朝向(法向量)
        std::vector<rerun::Position3D> normal_origins;  // 法向量起点

        // 解析 4 个装甲板的状态
        for(int i = 0; i < 4; ++i) {
            double x = robot.Armors(0, i);
            double y = robot.Armors(1, i);
            double z = robot.Armors(2, i);
            double theta = robot.Armors(3, i);
            
            // --- 核心修改：使用迎角投影判断物理可见性 ---
            Eigen::Matrix<double, 3, 1> P_i(x, y, z);
            double distance = P_i.norm();
            bool is_facing_us = false;
            
            if (distance > 1e-3) {
                // 1. 视线向量 (从枪管/相机指向装甲板)
                Eigen::Matrix<double, 3, 1> L_i = P_i / distance; 
                // 2. 装甲板法向量
                Eigen::Matrix<double, 3, 1> N_i(std::cos(theta), std::sin(theta), 0.0);
                
                // 3. 计算迎角投影
                double face_proj = -N_i.dot(L_i);
                
                // 如果投影 >= 0.17 (约夹角<=80度)，认为正面可见
                if (face_proj >= 0.17) {
                    is_facing_us = true;
                }
            }
            
            // 区分处于迎角可见范围内和处于盲区的装甲板
            if(is_facing_us) {
                visible_armors.push_back({(float)x, (float)y, (float)z});
            } else {
                hidden_armors.push_back({(float)x, (float)y, (float)z});
            }

            // 计算装甲板的朝向向量 (长度画成 15cm 方便观察)
            armor_normals.push_back({ (float)std::cos(theta) * 15.0f, (float)std::sin(theta) * 15.0f, 0.0f });
            normal_origins.push_back({(float)x, (float)y, (float)z});
        }

        // --- 渲染部分 ---

        // A. 渲染几何上朝向我们的装甲板 (亮红色点)
        if (!visible_armors.empty()) {
            rec.log("world/robot/armors_visible", 
                rerun::Points3D(visible_armors)
                    .with_colors({{255, 0, 0, 255}})
                    .with_radii({3.0f}));
        }
        
        // B. 渲染背对我们/侧偏太多的装甲板 (半透明灰色点)
        if (!hidden_armors.empty()) {
            rec.log("world/robot/armors_hidden", 
                rerun::Points3D(hidden_armors)
                    .with_colors({{150, 150, 150, 100}})
                    .with_radii({3.0f}));
        }

        // C. 渲染装甲板法向量 (白色箭头)
        rec.log("world/robot/orientation", 
            rerun::Arrows3D::from_vectors(armor_normals)
                .with_origins(normal_origins)
                .with_colors({{255, 255, 255, 200}}));

        // D. 渲染预测击打位置 (绿色点)
        std::vector<rerun::Position3D> predicted_pts;
        for(int i = 0; i < 4; ++i) {
            predicted_pts.push_back({ (float)aims(0, i), (float)aims(1, i), (float)aims(2, i) });
        }
        rec.log("world/robot/armors_predicted",
            rerun::Points3D(predicted_pts)
                .with_colors({{0, 255, 0, 255}}) // 纯绿色
                .with_radii({2.5f})
        );

        // E. 渲染机器人几何中心 (白色圆点)
        rec.log("world/robot/center",
            rerun::Points3D({{ (float)robot.center(0,0), (float)robot.center(1,0), (float)robot.center(2,0) }})
                .with_colors({{255, 255, 255, 255}})
                .with_radii({2.0f})
        );

        // F. 渲染线速度向量 (黄色大箭头) - 起点为中心点
        rec.log("world/robot/speed_vector",
            rerun::Arrows3D::from_vectors({{
                (float)robot.Speed(0, 0) , 
                (float)robot.Speed(1, 0) , 
                (float)robot.Speed(2, 0) 
            }})
            .with_origins({{
                (float)robot.center(0,0), 
                (float)robot.center(1,0), 
                (float)robot.center(2,0)
            }})
            .with_colors({{255, 255, 0, 255}})
        );

        // --- 新增：渲染当前枪管的瞄准射线 (青色长线) ---
        // 假设原点 {0,0,0} 为当前云台/相机位置，枪管是一根 20cm 长的射线
        rec.log("world/gun_ray",
            rerun::Arrows3D::from_vectors({{
                (float)Gun(0, 0) * 20.0f, 
                (float)Gun(1, 0) * 20.0f, 
                (float)Gun(2, 0) * 20.0f
            }})
            .with_origins({{0.0f, 0.0f, 0.0f}})
            .with_colors({{0, 255, 255, 255}}) // 青色 (Cyan)
        );

        // G. 渲染速度波形图 (Time Series Plots)
        // 1. 计算总速度: 提取线性速度的模长 (X,Y,Z 方向的综合大小)
        double total_speed = robot.Speed.norm();
        
        // 2. 获取旋转速度: 
        // ⚠️请注意：我不知道你的 Target.hpp 中旋转速度叫什么名字，这里假设叫 yaw_v
        // 如果你的名字是 robot.w 或者其他，请在这里修改！
        const double& rotation_speed = robot.Speed(3,0); // <--- TODO: 请替换为实际的角速度，如 robot.yaw_v;
        
        // 3. 记录绿线：总速度
        rec.log("world/speed/total_v", rerun::SeriesLines().with_colors({0, 255, 0, 255})); // 指定颜色为绿色
        rec.log("world/speed/total_v", rerun::Scalars((float)total_speed));                      // 传入数据

        // 4. 记录蓝线：旋转速度
        rec.log("world/speed/w", rerun::SeriesLines().with_colors({0, 0, 255, 255})); // 指定颜色为蓝色
        rec.log("world/speed/w", rerun::Scalars((float)rotation_speed));                   // 传入数据
    }

    void EKFKalmanUpdate(const Eigen::Matrix<double, 8, 1>& State,
                         const Eigen::Matrix<double, 4, 4>& CovArmor, 
                         const Eigen::Matrix<double, 4, 1>& View,
                         const Eigen::Matrix<double, 8, 8>& CovState,
                         const Eigen::Matrix<double, 8, 4>& KalmanGain, // 新增的卡尔曼增益
                         double radius, 
                         double dt)
    {
        // ==========================================
        // 1. 基础标量数据监控
        // ==========================================
        rec.log("EKF/dt", rerun::Scalars(dt));
        rec.log("EKF/radius", rerun::Scalars(radius));

        // ==========================================
        // 2. 3D 空间几何对比 (直观查看滤波效果)
        // ==========================================
        // 假设 State 结构为: [xc, yc, zc, yaw, v_xc, v_yc, v_zc, v_yaw]^T
        float xc = (float)State(0, 0);
        float yc = (float)State(1, 0);
        float zc = (float)State(2, 0);
        float w = (float)State(7, 0);

        

        // C. 当前实际观测到的目标位置 (青色点，假设 View 前三维是观测到的 x, y, z)
        rec.log("world/EKF/view_position", 
            rerun::Points3D({{(float)View(0, 0), (float)View(1, 0), (float)View(2, 0)}})
                .with_colors({{0, 255, 255, 255}}) // 青色
                .with_radii({3.0f}));
        rec.log("world/EKF/view_yaw",rerun::SeriesLines().with_colors({{0, 255, 255, 255}})); // 青色
        rec.log("world/EKF/view_yaw", 
            rerun::Scalars((float)View(3, 0))); // 青色

        
        // rec.log("world/EKF/w", rerun::SeriesLines().with_colors({{255, 165, 0, 255}})); // 橙色
        // rec.log("world/EKF/w", rerun::Scalars(w)); //

        // rec.log("world/EKF/v_total", rerun::SeriesLines().with_colors({{255, 165, 0, 255}})); // 橙色
        // rec.log("world/EKF/v_total", rerun::Scalars((float)State.block<3,1>(4,0).norm()/100.0f)); // 速度模长

        // ==========================================
        // 3. 协方差收敛波形图 (监控不确定性)
        // ==========================================
        // rec.log("plot/EKF_cov/pos_x_variance", rerun::SeriesLines().with_colors({255, 0, 0, 255}));
        // rec.log("plot/EKF_cov/pos_y_variance", rerun::SeriesLines().with_colors({0, 255, 0, 255}));
        // rec.log("plot/EKF_cov/yaw_variance",   rerun::SeriesLines().with_colors({0, 0, 255, 255}));

        // rec.log("plot/EKF_cov/pos_x_variance", rerun::Scalars(CovState(0, 0))); // x 方差
        // rec.log("plot/EKF_cov/pos_y_variance", rerun::Scalars(CovState(1, 1))); // y 方差
        // rec.log("plot/EKF_cov/yaw_variance",   rerun::Scalars(CovState(3, 3))); // yaw 方差

        // rec.log("plot/EKF_cov/measurement_x_var", rerun::SeriesLines().with_colors({255, 255, 255, 255}));
        // rec.log("plot/EKF_cov/measurement_x_var", rerun::Scalars(CovArmor(0, 0)));

        // ==========================================
        // 4. 卡尔曼增益 (Kalman Gain) 波形图
        // ==========================================
        // 假设 View(0) 是 X位置观测，State(0) 是 X位置状态，State(4) 是 X速度状态
        
        // A. 观测位置 X 对 状态位置 X 的增益 (K_xx)
        // rec.log("plot/EKF_gain/gain_posX_to_posX", rerun::SeriesLines().with_colors({255, 100, 100, 255}));
        // rec.log("plot/EKF_gain/gain_posX_to_posX", rerun::Scalars(KalmanGain(0, 0)));

        // // B. 观测位置 Y 对 状态位置 Y 的增益 (K_yy)
        // rec.log("plot/EKF_gain/gain_posY_to_posY", rerun::SeriesLines().with_colors({100, 255, 100, 255}));
        // rec.log("plot/EKF_gain/gain_posY_to_posY", rerun::Scalars(KalmanGain(1, 1)));

        // // C. 观测位置 X 对 状态速度 VX 的增益 (K_vx_x) 
        // // -> 这个值决定了当目标突然加速时，滤波器修正速度的灵敏度！
        // rec.log("plot/EKF_gain/gain_posX_to_velX", rerun::SeriesLines().with_colors({255, 200, 0, 255}));
        // rec.log("plot/EKF_gain/gain_posX_to_velX", rerun::Scalars(KalmanGain(4, 0)));
        
        // // D. 观测 yaw 对 状态 yaw_v 的增益 (如果你的模型包含角速度)
        // rec.log("plot/EKF_gain/gain_yaw_to_yawV", rerun::SeriesLines().with_colors({100, 100, 255, 255}));
        // rec.log("plot/EKF_gain/gain_yaw_to_yawV", rerun::Scalars(KalmanGain(7, 3))); 
    }


    void viewCov(const std::array<double, 4>& CovView) 
    {
        rec.log("Cov/x", rerun::Scalars(CovView[0]));
        rec.log("Cov/y", rerun::Scalars(CovView[1]));
        rec.log("Cov/z", rerun::Scalars(CovView[2]));
        rec.log("Cov/yaw", rerun::Scalars(CovView[3]));
    }
};
#endif // RERUN_VISUALIZER_HPP