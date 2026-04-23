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
        cv::Mat vis = frame.image.clone();

        // 绿色：CV 检测到的所有装甲板
        for (const auto& armor : opencv_armors) {
            std::vector<cv::Point> pts;
            for (const auto& p : armor.Lightcorners)
                pts.push_back({(int)p.x, (int)p.y});
            cv::polylines(vis, pts, true, cv::Scalar(0, 255, 0), 2);
        }

        // 红色：ResNet 识别结果，找回对应的 CVArmor 画框+标签
        for (const auto& armor : armors) {
            // 通过 center 匹配回 armors_2 的索引
            int idx = -1;
            for (size_t i = 0; i < armors_2.size(); i++) {
                if ((armor.center - armors_2[i][0].center).norm() < 1e-3 ||
                    (armor.center - armors_2[i][1].center).norm() < 1e-3) {
                    idx = (int)i;
                    break;
                }
            }
            if (idx < 0 || idx >= (int)opencv_armors.size()) continue;

            // 红框覆盖绿框
            std::vector<cv::Point> pts;
            for (const auto& p : opencv_armors[idx].Lightcorners)
                pts.push_back({(int)p.x, (int)p.y});
            cv::polylines(vis, pts, true, cv::Scalar(0, 0, 255), 2);

            // 标签
            cv::Point2f cen(0, 0);
            for (const auto& p : opencv_armors[idx].Lightcorners) cen += p;
            cen /= 4.0f;
            char label[32];
            std::snprintf(label, sizeof(label), "%s %.2f", typeName(armor.type), armor.confidence);
            cv::putText(vis, label, {(int)cen.x - 30, (int)cen.y - 12},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        // 左上角：追踪状态 + 模式
        {
            char buf[128];
            if (current_robot)
                std::snprintf(buf, sizeof(buf), "State:%s  Mode:%s",
                              stateName(track.getState()), modeName(current_robot->GetMode()));
            else
                std::snprintf(buf, sizeof(buf), "State:%s", stateName(track.getState()));
            cv::putText(vis, buf, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 255), 2);
        }

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
