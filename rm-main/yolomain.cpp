#include "Config.hpp"
#include "Function.hpp"
#include "YoloDetector.hpp"
#include "libxr.hpp"
#include "linux_uart.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

// #define MainDebug
#ifdef MainDebug
#include "communicate/RerunVisualizer.hpp"
RerunVisualizer viz("RoboMaster_AutoAim");
double R_sum = 0.0;
int R_count = 0;
#endif

namespace
{

constexpr const char* kConfigPath = "../../config/config.yaml";
constexpr const char* kModelPath = "../../model/yolo11.xml";
constexpr const char* kSerialDevice = "/dev/ttyACM0";
constexpr unsigned int kSerialBaud = 460800;

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
static FastQueue<std::unique_ptr<RobotState>> RobotStates(10);

std::chrono::steady_clock::time_point next_point = std::chrono::steady_clock::now();

io::HikCamera Hik(LoadHikCameraConfig(kConfigPath));

CVDetector opencv_detector(LoadCVDetectorConfig(kConfigPath));
YOLODetector yolo(kModelPath, YOLODetector::Camp::Blue);

Solver Sov(LoadSolverConfig(kConfigPath));
Tracker track(LoadRobotConfig(kConfigPath));
Shooter shoot(LoadShooterConfig(kConfigPath));
MPC::Planner planner(LoadPlannerConfig(kConfigPath));
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
    #ifdef MainDebug
    #ifdef EKFKalmanDebug
    g_ekf_debug_cb = [](const Eigen::Matrix<double,14,1>& s,
                        const Eigen::Matrix<double,4,1>& v, double dt) {
        viz.EKFKalmanUpdate(s, v, dt);
    };
    #endif
    #endif
    LibXR::PlatformInit();
    LibXR::RamFS ramfs;
    LibXR::LinuxUART uart(kSerialDevice, kSerialBaud);
    LibXR::HardwareContainer hw(
        LibXR::Entry<LibXR::LinuxUART>{uart, {"DevC-USB"}},
        LibXR::Entry<LibXR::RamFS>{ramfs, {"ramfs"}});
    LibXR::ApplicationManager appmgr;
    io::LibXRSerial<> ser(hw, appmgr);

    Hik.continueCap(3);

    std::thread match_thread([&]() { rm::IMUAndImageMatchFunction(Hik, ser, Frames); });
    std::thread plan_thread([&]() { rm::MPCPlanFunction(planner, RobotStates, ser, shoot); });

    std::printf("Start YOLO main loop\n");

    while (true) {
        FrameData frame;
        bool successpop = Frames.pop(frame);

        if(successpop){
            while (true) {
                bool ret = Frames.pop(frame);
                if (!ret) break;
            }
        }
        else{
            Frames.wait_pop(frame);
        }
        if (frame.image.empty()) continue;

        const Eigen::Matrix<double, 3, 1> Gun = shoot.GunDirection(frame.gripper_to_world);

        std::vector<ArmorPosi> armors = DetectArmors(frame.image, frame.gripper_to_world);

        double dt = rm::SolveDt(next_point, frame.time, 0.005);
        track(armors, frame.gripper_to_world, Gun, dt);
        next_point = frame.time;

        Robot* current_robot = track.getCurrentRobot();

        #ifdef MainDebug
            test.count();
        #endif

        if (current_robot == nullptr) {
            RobotStates.push(nullptr);
            continue;
        }

        RobotStates.push(std::make_unique<RobotState>(*current_robot, frame.time));

        #ifdef MainDebug
            viz.update(*current_robot, current_robot->Predict(0), dt, Gun);
        #endif
    }

    match_thread.join();
    plan_thread.join();
    return 0;
}
