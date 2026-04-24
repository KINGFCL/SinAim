#include "Demo.hpp"
#include "Config.hpp"

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

DemoReader demo("../../demo/damo.avi");
CVDetector detect(Light::Color::Blue);
ResNetNumClassifier resnet("../../model/tiny_resnet.onnx");
Solver::SolverConfig solver_config = LoadSolverConfig("../../config/solver.yaml");
Solver Sov(solver_config);
Tracker track;
Shooter shoot(0.005, 0.050);
Test test;

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
            if (cv::waitKey(1) == 27) break;
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
        auto opencv_armors = detect(frame.image, armors_pattern);

        Eigen::Quaterniond gripper_to_world{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z};
        std::vector<std::array<ArmorPosi, 2>> armors_2 = Sov(opencv_armors, gripper_to_world);

        // ── 分类 ────────────────────────────────────────────────
        std::vector<ArmorPosi> armors = resnet(armors_2, armors_pattern);

        // ── 追踪 ────────────────────────────────────────────────
        double dt = rm::SolveDt(next_point, frame.time, 0.01);
        track(armors, frame.quat, Gun, dt);
        next_point = frame.time;

        const auto& current_robot = track.getCurrentRobot();

        // ── 调试输出 ─────────────────────────────────────────────
        std::printf("[%d] CV:%zu Sov:%zu ResNet:%zu  State:%s",
                    frame_count,
                    opencv_armors.size(),
                    armors_2.size(),
                    armors.size(),
                    stateName(track.getState()));
        if (current_robot)
            std::printf("  Mode:%s", modeName(current_robot->GetMode()));
        if (!armors.empty()) {
            std::printf("  [");
            for (size_t i = 0; i < armors.size(); i++) {
                std::printf("%s(%.2f)%s", typeName(armors[i].type), armors[i].confidence,
                            i + 1 < armors.size() ? " " : "");
            }
            std::printf("]");
        }
        std::printf("\n"); std::fflush(stdout);

        // ── 可视化 ───────────────────────────────────────────────
        cv::Mat vis = rm::DrawSolverArmors(frame.image, armors_2, armors,
                                           frame.quat, solver_config,
                                           track.getState(), current_robot);

        cv::imshow("demo", vis);
        if (cv::waitKey(1) == 27) break;

        // ── 性能统计 ─────────────────────────────────────────────
        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();
        if (test.num % 200 == 0 && test.num != 0) { test.show(); test.clear(); }

        #ifdef MainDebug
        if (current_robot != nullptr)
            viz.update(*current_robot, current_robot->Predict(0), dt, Gun);
        #endif

        ++frame_count;
    }

    demo_thread.join();
    return 0;
}

void Test::count(const std::chrono::nanoseconds& time) { num++; total += time; }
void Test::clear() { num = 0; total = std::chrono::nanoseconds(0); }
void Test::show() { std::cout << num / ((double)total.count() * 1e-9) << "Hz\n"; }
