#include "Demo.hpp"
#include "Config.hpp"
#include "MlpNumClassifier.hpp"
#include "Target.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <vector>

#define MainDebug
#ifdef MainDebug
#include "communicate/RerunVisualizer.hpp"
RerunVisualizer viz;
double R_sum = 0.0;
int R_count = 0;
#endif



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

DemoReader demo("../../demo/damo.avi");
// ── 全局对象 ────────────────────────────────────────────────────
namespace
{
constexpr const char* kConfigPath = "../../config/config.yaml";
constexpr const char* kModelPath = "../../model/mlp.onnx";
constexpr const char* kSerialDevice = "/dev/ttyACM0";
constexpr unsigned int kSerialBaud = 460800;

struct Test
{
    int num = 0;
    std::chrono::nanoseconds total{0};

    void count(const std::chrono::nanoseconds& time)
    {
        ++num;
        total += time;
    }

    void clear()
    {
        num = 0;
        total = std::chrono::nanoseconds(0);
    }

    void show() const
    {
        std::cout << num / (static_cast<double>(total.count()) * 1e-9) << "Hz\n";
    }
};

static FastQueue<FrameData> Frames(10);
static FastQueue<std::unique_ptr<RobotState>> RobotStates(10);

std::chrono::steady_clock::time_point next_point = std::chrono::steady_clock::now();

io::RTSerial<Packet> ser(50);

CVDetector detect(LoadCVDetectorConfig(kConfigPath));
MlpNumClassifier mlp(kModelPath);

Solver Sov(LoadSolverConfig(kConfigPath));
Tracker track(LoadRobotConfig(kConfigPath));
Shooter shoot(LoadShooterConfig(kConfigPath));
MPC::Planner planner(LoadPlannerConfig(kConfigPath));
Test test;

std::vector<ArmorPosi> DetectArmors(cv::Mat& image, const Eigen::Quaterniond& gripper_to_world)
{
    std::vector<cv::Mat> armors_pattern;
    auto opencv_armors = detect(image, armors_pattern, CVDetector::ROIType::MLP);
    auto armors_2 = Sov(opencv_armors, gripper_to_world);
    return mlp(armors_2, armors_pattern);
}
}  // namespace

int main() {
    #ifdef EKFKalmanDebug
    g_ekf_debug_cb = [](const Eigen::Matrix<double,14,1>& s,
                        const Eigen::Matrix<double,4,1>& v, double dt) {
        viz.EKFKalmanUpdate(s, v, dt);
    };
    #endif

    if (!demo.isOpened()) {
        std::cerr << "Failed to open demo/damo.avi\n";
        return 1;
    }

    std::thread demo_thread(&DemoReader::feedQueue, &demo, std::ref(Frames));

    auto start = std::chrono::steady_clock::now();
    std::printf("Start main loop\n");

    int frame_count = 0;
    while (true)
    {
        if (Frames.empty()) {
            if (demo.isDone()) break;
            // if (cv::waitKey(1) == 27) break;
            continue;
        }

        FrameData frame;
        while (!Frames.empty())
            Frames.pop(frame);
        if (frame.image.empty()) continue;

        const Eigen::Matrix<double, 3, 1>& Gun =
            shoot.GunDirection(Eigen::Quaterniond{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z});

        // ── 检测 ────────────────────────────────────────────────
        std::vector<cv::Mat> armors_pattern;
        auto opencv_armors = detect(frame.image, armors_pattern, CVDetector::ROIType::MLP);

        Eigen::Quaterniond gripper_to_world{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z};
        std::vector<std::array<ArmorPosi, 2>> armors_2 = Sov(opencv_armors, gripper_to_world);

        // ── 分类 ────────────────────────────────────────────────
        std::vector<ArmorPosi> armors = mlp(armors_2, armors_pattern);
        // std::cout<<"----------------------------------\n一帧的数据: ";
        // for(auto armor_:armors)
        // {
        //     std::cout<< "yaw0_abs: " << armor_.yaw_abs[0] << " yaw1_abs: " << armor_.yaw_abs[1] << 
        //     " yaw0: " << armor_.yaw[0] << " yaw1: " << armor_.yaw[1] << " armor.center_theta0: "<< armor_.theta[0] << " armor.center_theta1: "
        //        << armor_.theta[1]<<"\n"<<armor_.center<< " err0: " << armor_.reproj[0] << 
        //        " err1: " << armor_.reproj[1] ;

        // }
        // ── 追踪 ────────────────────────────────────────────────
        double dt = rm::SolveDt(next_point, frame.time, 0.01);
        track(armors, frame.quat, Gun, dt);
        next_point = frame.time;

        const auto& current_robot = track.getCurrentRobot();

        // ── 调试输出 ─────────────────────────────────────────────
        // std::printf("[%d] CV:%zu Sov:%zu ResNet:%zu  State:%s",
        //             frame_count,
        //             opencv_armors.size(),
        //             armors_2.size(),
        //             armors.size(),
        //             stateName(track.getState()));
        if (current_robot)
            //std::printf("  Mode:%s", modeName(current_robot->GetMode()));
        if (!armors.empty()) {
            // std::printf("  [");
            // for (size_t i = 0; i < armors.size(); i++) {
            //     std::printf("%s(%.2f)%s", typeName(armors[i].type), armors[i].confidence,
            //                 i + 1 < armors.size() ? " " : "");
            // }
            //std::printf("]");
        }
        //std::printf("\n"); std::fflush(stdout);

        // // ── 可视化 ───────────────────────────────────────────────
        // cv::Mat vis = rm::DrawSolverArmors(frame.image, armors_2, armors,
        //                                    frame.quat, solver_config,
        //                                    track.getState(), current_robot);

        // cv::imshow("demo", vis);
        // if (cv::waitKey(1) == 27) break;

        // ── 性能统计 ─────────────────────────────────────────────
        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();
        if (test.num % 200 == 0 && test.num != 0) { test.show(); test.clear(); }

        #ifdef MainDebug
        if (current_robot != nullptr)
            if(current_robot->GetMode() == Robot::KalmanMode::EKF)
                viz.update(*current_robot, current_robot->Predict(0), dt, Gun);
        #endif

        ++frame_count;
    }

    demo_thread.join();
    return 0;
}


