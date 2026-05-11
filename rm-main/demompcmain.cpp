#include "Demo.hpp"
#include "Config.hpp"
#include "MlpNumClassifier.hpp"
#include "Target.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <thread>
#include <vector>

#define MainDebug
#ifdef MainDebug
#include "communicate/RerunVisualizer.hpp"
RerunVisualizer viz;
double R_sum = 0.0;
int R_count = 0;
#endif

DemoReader demo("../../demo/damo.avi");

namespace
{
constexpr const char* kConfigPath = "../../config/config.yaml";
constexpr const char* kModelPath = "../../model/mlp.onnx";

struct Test
{
    explicit Test(int cycle):cycle(cycle){};
private:
    int num = 0;
    const int cycle = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
public:
    void count()
    {
        ++this->num;
        if(this->num == this->cycle)
        {
            this->show();
            this->clear();
        }
        
    }

    void clear()
    {
        num = 0;
        this->start = std::chrono::steady_clock::now();
    }

    void show() const
    {
        auto total = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - this->start);
        std::cout << num / (static_cast<double>(total.count()) * 1e-9) << "Hz\n";
    }
};

static FastQueue<FrameData> Frames(10);

std::chrono::steady_clock::time_point next_point = std::chrono::steady_clock::now();

CVDetector detect(LoadCVDetectorConfig(kConfigPath));
MlpNumClassifier mlp(kModelPath);

Solver Sov(LoadSolverConfig(kConfigPath));
Tracker track(LoadRobotConfig(kConfigPath));
Shooter shoot(LoadShooterConfig(kConfigPath));
Test test(200);

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

    std::printf("Start demo MLP main loop\n");

    while (true)
    {
        FrameData frame;
        bool successpop = Frames.pop(frame);

        if(successpop){
            while (true) {
                bool ret = Frames.pop(frame);
                if (!ret) break;
            }
        }
        else{
            if (demo.isDone()) break;
            if (!Frames.wait_pop(frame, std::chrono::milliseconds(10))) continue;
        }
        if (frame.image.empty()) continue;

        // auto a1 = std::chrono::steady_clock::now();
        Eigen::Quaterniond gripper_to_world{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z};
        const Eigen::Matrix<double, 3, 1> Gun = shoot.GunDirection(gripper_to_world);

        std::vector<ArmorPosi> armors = DetectArmors(frame.image, gripper_to_world);

        double dt = rm::SolveDt(next_point, frame.time, 0.005);
        track(armors, frame.quat, Gun, dt);
        next_point = frame.time;

        Robot* current_robot = track.getCurrentRobot();
        // std::cout << (std::chrono::steady_clock::now()-a1).count()*1e-6<< "ms\n";
        #ifdef MainDebug
            test.count();
        #endif

        if (current_robot == nullptr) {
            continue;
        }

        #ifdef MainDebug
            viz.update(*current_robot, current_robot->Predict(0), dt, Gun);
        #endif
    }

    demo_thread.join();
    return 0;
}

