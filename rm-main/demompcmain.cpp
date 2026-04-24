#include "Demo.hpp"
#include "include/Detector.hpp"
#include "include/Solver.hpp"
#include "include/SmallNumClassifier.hpp"
#include "include/Shooter.hpp"
#include "include/Tracker.hpp"
#include "include/RerunVisualizer.hpp"

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

static FastQueue<FrameData> Frames(10);
std::chrono::steady_clock::time_point next_point = std::chrono::steady_clock::now();

DemoReader demo("../../demo/damo.avi");
Detector detect(Light::Color::Blue, 0.4);
SmallNumClassifier smallnet("../model/mlp.onnx");

Solver Sov("../../config/Solver_config.yaml");
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
            continue;
        }

        FrameData frame;
        while (!Frames.empty())
            Frames.pop(frame);
        if (frame.image.empty()) continue;

        const Eigen::Matrix<double, 3, 1>& Gun =
            shoot.GunDirection(Eigen::Quaterniond{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z});

        // 检测
        std::vector<cv::Mat> armors_pattern;
        auto opencv_armors = detect(frame.image, armors_pattern,true);;

        // 解算
        std::vector<std::array<ArmorPosi, 2>> armors_2 = Sov(opencv_armors);
        Sov.Filter(armors_2, armors_pattern, frame.quat, Gun);

        // 分类
        std::vector<ArmorPosi> armors = smallnet(armors_2, armors_pattern);

        std::cout<<"----------------------------------\n一帧的数据: ";
        for(auto armor_:armors)
        {
            std::cout<< " theta: " << armor_.theta
                     << " posi: (" << armor_.posi.x << ", " << armor_.posi.y << ", " << armor_.posi.z << ")"
                     << " error: " << armor_.error << "\n";
        }

        // 坐标系转换
        Sov.ConverToWorld(armors, frame.quat);

        // 追踪
        double dt = rm::SolveDt(next_point, frame.time, 0.01);
        track(armors, frame.quat, Gun, dt);
        next_point = frame.time;

        const auto& current_robot = track.getCurrentRobot();

        // 性能统计
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
