#include "Solver.hpp"
#include "Demo.hpp"
#include "Config.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <vector>

#define MainDebug
#ifdef MainDebug
#include "communicate/RerunVisualizer.hpp"
RerunVisualizer viz("RoboMaster_AutoAim");
double R_sum = 0.0;
int R_count = 0;
#endif

struct Test
{
    int num = 0;
    std::chrono::nanoseconds total{0};
    void count(const std::chrono::nanoseconds& time);
    void clear();
    void show();
};

// ── 辅助：枚举转字符串 ──────────────────────────────────────────
static const char* typeName(ArmorPosi::Type t) {
    switch (t) {
        case ArmorPosi::Type::hero:    return "hero";
        case ArmorPosi::Type::two:     return "2";
        case ArmorPosi::Type::three:   return "3";
        case ArmorPosi::Type::four:    return "4";
        case ArmorPosi::Type::guard:   return "guard";
        case ArmorPosi::Type::outpost: return "outpost";
        case ArmorPosi::Type::base:    return "base";
        default:                       return "?";
    }
}
static const char* stateName(Tracker::State s) {
    switch (s) {
        case Tracker::State::Searching: return "Searching";
        case Tracker::State::Tracking:  return "Tracking";
        case Tracker::State::TempLost:  return "TempLost";
        case Tracker::State::Lost:      return "Lost";
        default:                        return "?";
    }
}
static const char* modeName(Robot::KalmanMode m) {
    return m == Robot::KalmanMode::EKF ? "EKF" : "LKF";
}

// ── 全局对象 ────────────────────────────────────────────────────
static FastQueue<FrameData> Frames(10);
std::chrono::steady_clock::time_point next_point = std::chrono::steady_clock::now();
cv::VideoCapture cap_("../../demo/damo.avi");

CVDetector detect(Light::Color::Blue);
ResNetNumClassifier resnet("../../model/tiny_resnet.onnx");
Solver::SolverConfig solver_config = LoadSolverConfig("../../config/solver.yaml");
Solver Sov(solver_config);
Test test;

int main() {
    #ifdef EKFKalmanDebug
    g_ekf_debug_cb = [](const Eigen::Matrix<double,14,1>& s,
                        const Eigen::Matrix<double,4,1>& v, double dt) {
        viz.EKFKalmanUpdate(s, v, dt);
    };
    #endif




    auto start = std::chrono::steady_clock::now();
    std::printf("Start main loop\n");

    int frame_count = 0;
    while (true)
    {

        cv::Mat raw;
        cap_.read(raw);
        if (raw.empty()) continue;
        struct DoubleQuaternion { double time, w, x, y, z; } dq;
        std::memcpy(&dq, raw.data, sizeof(dq));
        FrameData frame(raw, cv::Quatd(dq.w, dq.x, dq.y, dq.z), std::chrono::steady_clock::now());


        // ── 检测 ────────────────────────────────────────────────
        std::vector<cv::Mat> armors_pattern;
        auto opencv_armors = detect(frame.image, armors_pattern);
        cv::Mat detect_show = frame.image.clone();
        detect.ArmorShow(detect_show, opencv_armors);
        cv::imshow("dtecter",detect_show);

        Eigen::Quaterniond gripper_to_world{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z};
        

        std::vector<std::array<ArmorPosi, 2>> armors_2 = Sov(opencv_armors, gripper_to_world);

        // ── 分类 ────────────────────────────────────────────────
        std::vector<ArmorPosi> armors = resnet(armors_2, armors_pattern);
        // std::cout<<"----------------------------------\n一帧的数据: ";
        // for(auto armor_:armors)
        // {
        //     std::cout<< "yaw0_abs: " << armor_.yaw_abs[0] << " yaw1_abs: " << armor_.yaw_abs[1] << 
        //     " yaw0: " << armor_.yaw[0] << " yaw1: " << armor_.yaw[1] << " armor.center_theta0: "<< armor_.theta[0] << " armor.center_theta1: "
        //        << armor_.theta[1]<<"\n"<<armor_.center<< " err0: " << armor_.reproj[0] << 
        //        " err1: " << armor_.reproj[1] ;

        // }
        // ── 追踪 ────────────────────────────────────────────────

        if(!armors.empty())
        {
            viz.show("raw_show_yaw", std::remainder(armors[0].yaw + M_PI, 2 * M_PI));
        }
        // ── 可视化：重投影角点 ────────────────────────────────────────
        {
            cv::Mat vis = frame.image.clone();

            cv::Mat camMat(3, 3, CV_64FC1, const_cast<double*>(solver_config.camera_matrix.data()));
            cv::Mat distC(5, 1, CV_64FC1, const_cast<double*>(solver_config.distortion_coeffs.data()));

            // yaw = atan2(width_world.y, width_world.x)，即装甲板X轴（宽度方向）在世界系的yaw
            // 与旧 solveRectanglePose 的法线yaw不同，这里直接用宽度方向重建旋转矩阵
            auto ArmorPosiToCam = [](double yaw, Eigen::Matrix3d R_cam2world, Eigen::Vector3d center) -> Eigen::Matrix<double,3,4>
            {
                // 装甲板X轴（宽度方向）
                Eigen::Vector3d x_w(std::cos(yaw), std::sin(yaw), 0.0);
                // 世界Z轴（竖直向上）在装甲板X轴垂直平面内的分量作为Y轴（高度方向）
                Eigen::Vector3d z_up(0.0, 0.0, 1.0);
                Eigen::Vector3d y_w = (z_up - z_up.dot(x_w) * x_w).normalized();
                // Z轴 = 法线方向
                Eigen::Vector3d z_w = x_w.cross(y_w);
                Eigen::Matrix3d R_world2base;
                R_world2base.col(0) = x_w;
                R_world2base.col(1) = y_w;
                R_world2base.col(2) = z_w;
                // object points 与 solvePnP 一致：XY平面，Z=0
                double W = 13.5, H = 5.5;
                const Eigen::Matrix<double,3,4> P {
                    {-W*0.5, W*0.5,  W*0.5, -W*0.5},
                    {-H*0.5, -H*0.5, H*0.5,  H*0.5},
                    {0.0,    0.0,    0.0,    0.0   }
                };
                Eigen::Matrix<double,3,4> P_world = R_world2base * P;
                P_world = P_world.colwise() + center;
                return R_cam2world.transpose() * P_world;
            };

            Eigen::Matrix3d R_c2g = Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>>(solver_config.R_Cam_to_gripper.data());
            Eigen::Vector3d T_c2g = Eigen::Map<const Eigen::Vector3d>(solver_config.T_Cam_to_gripper.data());
            Eigen::Matrix3d R_cam2world = gripper_to_world.toRotationMatrix() * R_c2g;
            Eigen::Vector3d photocenter = gripper_to_world.toRotationMatrix() * T_c2g;

            auto drawCorners = [&](const Eigen::Vector3d& cen, double yaw, const cv::Scalar& color) {
                Eigen::Matrix<double,3,4> corners_cam = ArmorPosiToCam(yaw, R_cam2world, cen - photocenter);
                for (int i = 0; i < 4; ++i) {
                    Eigen::Vector3d c = corners_cam.col(i);
                    std::vector<cv::Point3f> obj{{(float)c.x(), (float)c.y(), (float)c.z()}};
                    std::vector<cv::Point2f> img;
                    cv::projectPoints(obj, cv::Vec3d(0,0,0), cv::Vec3d(0,0,0), camMat, distC, img);
                    cv::circle(vis, img[0], 5, color, -1);
                }
            };

            // 绿色：原始检测角点
            for (const auto& oa : opencv_armors)
                for (const auto& pt : oa.Lightcorners)
                    cv::circle(vis, pt, 5, cv::Scalar(0, 255, 0), -1);

            // 小装甲：解0红，解1蓝；大装甲：解0橙，解1紫
            for (const auto& pair : armors_2) {
                const auto& sm = pair[0];
                const auto& bg = pair[1];
                if (sm.IsInRange) {
                    drawCorners(sm.center, std::remainder(sm.yaw + M_PI, 2*M_PI), cv::Scalar(255, 0, 0));
                }
                if (false) {
                    drawCorners(bg.center, bg.yaw, cv::Scalar(0, 165, 255));
                }
            }

            cv::imshow("demo", vis);
            if (cv::waitKey(0) == 27) break;
        }

        // ── 性能统计 ─────────────────────────────────────────────
        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();
        if (test.num % 200 == 0 && test.num != 0) { test.show(); test.clear(); }


        ++frame_count;
    }

    return 0;
}

void Test::count(const std::chrono::nanoseconds& time) { num++; total += time; }
void Test::clear() { num = 0; total = std::chrono::nanoseconds(0); }
void Test::show() { std::cout << num / ((double)total.count() * 1e-9) << "Hz\n"; }
