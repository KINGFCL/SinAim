#include "Config.hpp"
#include "Function.hpp"
#include "YoloDetector.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace
{
constexpr const char* kConfigPath = "../../config/config.yaml";
constexpr const char* kModelPath = "../../model/yolo11.xml";
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
YOLODetector yolo(kModelPath, YOLODetector::Camp::Blue);
io::HikCamera Hik(LoadHikCameraConfig(kConfigPath));


Solver Sov(LoadSolverConfig(kConfigPath));
Tracker track(LoadRobotConfig(kConfigPath));
Shooter shoot(LoadShooterConfig(kConfigPath));
MPC::Planner planner(LoadPlannerConfig(kConfigPath));
Test test;

std::vector<ArmorPosi> DetectArmors(cv::Mat& image, const Eigen::Quaterniond& gripper_to_world)
{
    std::vector<YoloArmor> yolo_armors = yolo(image);
    return Sov(yolo_armors, gripper_to_world);
}
}  // namespace

int main()
{
    std::cout << sizeof(Packet) << std::endl;

    std::function<bool(const Packet&)> check_fuc = io::CRC8::Check<Packet>;
    ser.setCheckfuc(check_fuc);
    int ret = ser.openDevice(kSerialDevice, kSerialBaud);

    if (ret == 1) {
        std::cout << "serial open ok\n";
    } else {
        std::cerr << "serial open err: " << ret << "\n";
    }

    ser.startReceive(100);
    Hik.continueCap(3);

    std::thread match_thread(rm::IMUAndImageMatchFunction, std::ref(Hik), std::ref(ser), std::ref(Frames));
    std::thread plan_thread([&]() { rm::MPCPlanFunction(planner, RobotStates, ser, shoot); });

    auto start = std::chrono::steady_clock::now();
    std::printf("Start YOLO main loop\n");

    while (true) {
        if (Frames.empty()) continue;

        FrameData frame;
        while (!Frames.empty()) {
            Frames.pop(frame);
        }
        if (frame.image.empty()) continue;

        Eigen::Quaterniond gripper_to_world{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z};
        const Eigen::Matrix<double, 3, 1> Gun = shoot.GunDirection(gripper_to_world);

        std::vector<ArmorPosi> armors = DetectArmors(frame.image, gripper_to_world);

        double dt = rm::SolveDt(next_point, frame.time, 0.005);
        track(armors, frame.quat, Gun, dt);
        next_point = frame.time;

        Robot* current_robot = track.getCurrentRobot();
        if (current_robot == nullptr) {
            RobotStates.push(nullptr);
            continue;
        }

        RobotStates.push(std::make_unique<RobotState>(*current_robot, frame.time));

        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();

        if (test.num % 200 == 0 && test.num != 0) {
            test.show();
            test.clear();
        }
    }

    match_thread.join();
    plan_thread.join();
    return 0;
}
