// 示例主程序
#include <iostream>
#include <opencv2/opencv.hpp>
#include "HikCamera.hpp"
#include "CVDetector.hpp"
#include "ResNetNumClassifier.hpp"
#include "YoloDetector.hpp"
#include "Solver.hpp"
#include "Tracker.hpp"
#include "planner.hpp"
#include "RTSerial.hpp"
#include "Data.hpp"

#include "Shooter.hpp"
#include "FastQueue.hpp"

    static FastQueue<std::unique_ptr<RobotState>> robotStates(10);

int main() {
    // 初始化各模块
    io::HikCamera Hik(1,16);
    io::RTSerial<Packet> serial;
    CVDetector detector(Light::Color::Red, cv::Size(32, 32));
    YOLODetector yolo("model.xml", YOLODetector::Camp::Red);
    ResNetNumClassifier resnet("model.onnx", 0.5f);
    //Solver solver("config.yaml");
    Tracker tracker;
    MPC::Planner planner("config.yaml");
    Shooter shoot(0.005,0.050);


    FrameData frame;
    std::vector<cv::Mat> armors_pattern;
    std::deque<CVArmor> armors = detector(frame.image, armors_pattern);

    //std::vector<std::array<ArmorPosi, 2>> armors_posis = solver(armors);



    //solver.ConverToWorld(armors_posis, cv::Quatd{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z});

    // hax = pose_detector.InWorld(armors_posis, armors_pattern, hax, frame.quat);

    // std::vector<ArmorPosi> classified_armors = resnet(armors_posis, armors_pattern, hax);

    auto Gun = shoot.GunDirection(Eigen::Quaterniond(frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z));

    // tracker(classified_armors, frame.quat, Gun, 0.01);

    //     const auto& current_robot = tracker.getCurrentRobot();

    //     if (current_robot == nullptr)
    //     {
    //         // 没有追踪到目标，跳过
    //         robotStates.push(nullptr);
    //         //continue;
    //     }

    //     robotStates.push(std::make_unique<RobotState>(*current_robot,frame.time));

    std::cout << "RoboMaster AutoAim System" << std::endl;
    return 0;
}
