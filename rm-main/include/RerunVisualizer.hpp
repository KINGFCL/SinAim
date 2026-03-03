#ifndef RERUN_VISUALIZER_HPP
#define RERUN_VISUALIZER_HPP

#include <rerun.hpp>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <vector>
#include <string>

// 包含你的 Robot 定义
#include "Target.hpp" 

class RerunVisualizer {
private:
    rerun::RecordingStream rec;

public:
    // 构造函数：初始化 Rerun 并唤起 Viewer
    RerunVisualizer(const std::string& app_name = "RoboMaster_AutoAim") 
        : rec(app_name) 
    {
        // 尝试自动启动独立的 Rerun Viewer 窗口
        rec.spawn().exit_on_failure();
    }

    // 析构函数
    ~RerunVisualizer() = default;

    /**
     * @brief 核心更新函数，在 main 的 while 循环中调用
     * @param robot 当前机器人的状态 (包含装甲板、速度、中心点)
     * @param aims  预测出的击打点矩阵 (4x4)
     * @param dt    预测时间
     * @param image 相机原图 (可选，用于同步显示2D画面)
     */
    void update(const Robot& robot, const Eigen::Matrix<double, 4, 4>& aims, double dt, const cv::Mat& image) 
    {
        // 1. 记录预测时间 dt (在 Viewer 中生成时间折线图)
        rec.log("debug/dt", rerun::Scalar(dt));

        // 2. 准备 3D 渲染所需的数据容器
        std::vector<rerun::Position3D> visible_armors;
        std::vector<rerun::Position3D> hidden_armors;
        std::vector<rerun::Position3D> armor_normals;   // 装甲板朝向(法向量)
        std::vector<rerun::Position3D> normal_origins;  // 法向量起点

        // 解析 4 个装甲板的状态
        for(int i = 0; i < 4; ++i) {
            float x = robot.Armors(0, i);
            float y = robot.Armors(1, i);
            float z = robot.Armors(2, i);
            float theta = robot.Armors(3, i);
            
            // 区分真实视野可见和卡尔曼推算的装甲板
            if(robot.View[i] == Robot::ArmorView::Visual) {
                visible_armors.push_back({x, y, z});
            } else {
                hidden_armors.push_back({x, y, z});
            }

            // 计算装甲板的朝向向量 (长度画成 15cm 方便观察)
            armor_normals.push_back({ std::cos(theta) * 15.0f, std::sin(theta) * 15.0f, 0.0f });
            normal_origins.push_back({x, y, z});
        }

        // --- 渲染部分 ---

        // A. 渲染真实视觉可见的装甲板 (亮红色点)
        if (!visible_armors.empty()) {
            rec.log("world/robot/armors_visible", 
                rerun::Points3D(visible_armors)
                    .with_colors({{255, 0, 0, 255}})
                    .with_radii({3.0f}));
        }
        
        // B. 渲染盲区推算的装甲板 (半透明灰色点)
        if (!hidden_armors.empty()) {
            rec.log("world/robot/armors_hidden", 
                rerun::Points3D(hidden_armors)
                    .with_colors({{150, 150, 150, 100}})
                    .with_radii({3.0f}));
        }

        // C. 渲染装甲板朝向指示 (白色箭头)
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
        // 乘以 15.0f 是为了放大速度矢量，否则在 3D 空间看不清
        rec.log("world/robot/speed_vector",
            rerun::Arrows3D::from_vectors({{
                (float)robot.Speed(0, 0) * 15.0f, 
                (float)robot.Speed(1, 0) * 15.0f, 
                (float)robot.Speed(2, 0) * 15.0f
            }})
            .with_origins({{
                (float)robot.center(0,0), 
                (float)robot.center(1,0), 
                (float)robot.center(2,0)
            }})
            .with_colors({{255, 255, 0, 255}})
        );

        // G. (可选) 同步渲染 2D 图像画面
        // 这样你可以直接删掉原代码里的 cv::imshow("frame", frame.image); 和 cv::waitKey(1);
        if (!image.empty()) {
            // Rerun 0.16 及以上版本支持直接从 BGR 数据加载
            rec.log("camera/image", rerun::Image::from_bgr8(
                image.data, 
                {static_cast<size_t>(image.cols), static_cast<size_t>(image.rows)}
            ));
        }
    }
};

#endif // RERUN_VISUALIZER_HPP