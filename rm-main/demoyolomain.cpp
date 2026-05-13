#include "Demo.hpp"
#include "Config.hpp"
#include "Target.hpp"
#include "YoloDetector.hpp"

#include <chrono>
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
constexpr const char* kModelPath = "../../model/yolo11.xml";

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

YOLODetector yolo(kModelPath, YOLODetector::Camp::Blue, 0.5f, 0.2f, "AUTO");
CVDetector opencv_detector(LoadCVDetectorConfig(kConfigPath));
Solver Sov(LoadSolverConfig(kConfigPath));
Tracker track(LoadRobotConfig(kConfigPath));
Shooter shoot(LoadShooterConfig(kConfigPath));
Test test(200);

std::vector<ArmorPosi> DetectArmors(cv::Mat& image, const Eigen::Quaterniond& gripper_to_world)
{
    opencv_detector.rgb_img = image;
    cv::Mat binary_img = opencv_detector.preprocessImage(image);
    std::vector<CVArmor> opencv_armors = opencv_detector.FindArmor(opencv_detector.FindLight(binary_img));

    std::vector<YoloArmor> yolo_armors = yolo(image);
    std::vector<YoloArmor> fused_armors = rm::MatchYoloAndOpenCV(opencv_armors, yolo_armors);

    return Sov(fused_armors, gripper_to_world);
}
}  // namespace

int main()
{
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

    std::printf("Start demo YOLO main loop\n");

    while (true)
    {
        FrameData frame;
        bool successpop = Frames.pop(frame);

        if (successpop) {
            while (true) {
                bool ret = Frames.pop(frame);
                if (!ret) break;
            }
        } else {
            if (demo.isDone()) break;
            if (!Frames.wait_pop(frame, std::chrono::milliseconds(10))) continue;
        }

        if (frame.image.empty()) continue;


        const Eigen::Matrix<double, 3, 1> Gun = shoot.GunDirection(frame.gripper_to_world);

        std::vector<ArmorPosi> armors = DetectArmors(frame.image, frame.gripper_to_world);

        double dt = rm::SolveDt(next_point, frame.time, 0.005);
        track(armors, frame.gripper_to_world, Gun, dt);
        next_point = frame.time;

        Robot* current_robot = track.getCurrentRobot();
        OutPust* current_outpust = track.getCurrentOutPust();

#ifdef MainDebug
        test.count();
#endif

        if (current_robot == nullptr && current_outpust == nullptr) {
            continue;
        }

#ifdef MainDebug
        if (current_robot != nullptr) {
        viz.update(*current_robot, current_robot->Predict(0), dt, Gun);
        }
#endif
    }

    demo_thread.join();
    return 0;
}
