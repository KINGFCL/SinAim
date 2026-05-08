#include "Config.hpp"
#include "Function.hpp"
#include "YoloDetector.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>
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

io::HikCamera Hik(1, 16);
io::RTSerial<Packet> ser(50);

YOLODetector yolo(kModelPath, YOLODetector::Camp::Blue);

Solver Sov(LoadSolverConfig(kConfigPath));
Tracker track;
Shooter shoot(LoadShooterConfig(kConfigPath));
MPC::Planner planner(LoadPlannerConfig(kConfigPath));
Test test;

Light MakeLightFromEdge(const cv::Point2f& top, const cv::Point2f& bottom)
{
    cv::Point2f center = (top + bottom) * 0.5f;
    cv::Point2f edge = bottom - top;
    float length = std::max(1.0f, static_cast<float>(cv::norm(edge)));
    float width = std::max(2.0f, length * 0.1f);
    float edge_angle = static_cast<float>(std::atan2(edge.y, edge.x) * 180.0 / 3.14159265358979323846);
    return Light(cv::RotatedRect(center, cv::Size2f(width, length), edge_angle - 90.0f));
}

CVArmor ToCVArmor(const YoloArmor& armor)
{
    Light left = MakeLightFromEdge(armor.keypoints[0], armor.keypoints[3]);
    Light right = MakeLightFromEdge(armor.keypoints[1], armor.keypoints[2]);
    CVArmor cv_armor(left, right);
    cv_armor.Lightcorners = armor.keypoints;
    return cv_armor;
}

std::pair<ArmorPosi::Type, int> DecodeYoloClass(int class_id)
{
    if (class_id >= 0 && class_id <= 2) {
        return {ArmorPosi::Type::guard, 0};
    }
    if (class_id >= 3 && class_id <= 5) {
        return {ArmorPosi::Type::hero, 1};
    }
    if (class_id >= 6 && class_id <= 8) {
        return {ArmorPosi::Type::two, 0};
    }
    if (class_id >= 9 && class_id <= 11) {
        return {ArmorPosi::Type::three, 0};
    }
    if (class_id >= 12 && class_id <= 14) {
        return {ArmorPosi::Type::four, 0};
    }
    if (class_id >= 18 && class_id <= 20) {
        return {ArmorPosi::Type::outpost, 0};
    }
    if (class_id >= 21 && class_id <= 28) {
        return {ArmorPosi::Type::base, 1};
    }
    if (class_id >= 29 && class_id <= 31) {
        return {ArmorPosi::Type::three, 0};
    }
    if (class_id >= 32 && class_id <= 34) {
        return {ArmorPosi::Type::four, 0};
    }
    return {ArmorPosi::Type::Unknow, 0};
}

std::vector<ArmorPosi> DetectArmors(cv::Mat& image, const Eigen::Quaterniond& gripper_to_world)
{
    std::vector<YoloArmor> yolo_armors = yolo(image);
    std::vector<YoloArmor> valid_yolo_armors;
    std::vector<CVArmor> cv_armors;

    valid_yolo_armors.reserve(yolo_armors.size());
    cv_armors.reserve(yolo_armors.size());

    for (const auto& armor : yolo_armors) {
        if (armor.keypoints.size() != 4) continue;

        valid_yolo_armors.push_back(armor);
        cv_armors.push_back(ToCVArmor(armor));
    }

    auto armors_2 = Sov(cv_armors, gripper_to_world);

    std::vector<ArmorPosi> armors;
    armors.reserve(armors_2.size());

    for (size_t i = 0; i < armors_2.size() && i < valid_yolo_armors.size(); ++i) {
        auto [type, idx] = DecodeYoloClass(valid_yolo_armors[i].class_id);
        if (type == ArmorPosi::Type::Unknow) continue;
        if (!armors_2[i][idx].IsInRange) continue;

        armors.push_back(armors_2[i][idx]);
        armors.back().type = type;
        armors.back().confidence = valid_yolo_armors[i].conf;
    }

    return armors;
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
    std::thread plan_thread(rm::MPCPlanFunction, std::ref(planner), std::ref(RobotStates), std::ref(ser));

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
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            rm::SendMessageToRobot(ser, 0.0, 0.0, false);
            continue;
        }

        if (current_robot->GetMode() == Robot::KalmanMode::EKF) {
            RobotStates.push(std::make_unique<RobotState>(*current_robot, frame.time));
        } else {
            RobotStates.push(nullptr);
            double fly_time = shoot.FlyTime(current_robot->center);
            Eigen::Vector3d aim = current_robot->center + current_robot->Speed.block<3, 1>(0, 0) * fly_time;
            std::array<double, 2> pitch_and_yaw = shoot(aim);
            rm::SendMessageToRobot(ser, pitch_and_yaw[0], pitch_and_yaw[1], true);
        }

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
