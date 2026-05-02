#include "Function.hpp"
#include "aim/planner/planner.hpp"
#include "Config.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Core>
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <ratio>
#include <thread>
#include <vector>

// #define MainDebug
#ifdef MainDebug
#include "communicate/RerunVisualizer.hpp"
RerunVisualizer viz("RoboMaster_AutoAim");
double R_sum = 0.0;
int R_count = 0;
#endif

//性能测试工具
struct Test
{
    int num = 0;
    std::chrono::nanoseconds total{0};

    void count(const std::chrono::nanoseconds& time);
    void clear();
    void show();
};

static FastQueue<FrameData> Frames(10);
static FastQueue<std::unique_ptr<RobotState>> robotStates(10);

std::chrono::steady_clock::time_point next_point = std::chrono::steady_clock::now();

io::HikCamera Hik(1,16);
io::RTSerial<Packet> ser(50);

// 传统视觉检测器
CVDetector detect(Light::Color::Blue);

// 数字分类器
ResNetNumClassifier resnet("../../model/tiny_resnet.onnx");



Solver::SolverConfig solver_config = LoadSolverConfig("../../config/solver.yaml");
Solver Sov(solver_config);

// 追踪器
Tracker track;

// Shooter shoot(Eigen::Matrix<double,3,1>(0.9999283656297303,
//  0.002822337907989043,
//  -0.01163176761242803));
Shooter shoot(0.005,0.050);

MPC::Planner planner("../../config/planner.yaml");
Test test;

int main() {

    //1.0初始化串口
    std::cout<<sizeof(Packet)<<std::endl;

    std::function<bool(const Packet&)> check_fuc = io::CRC8::Check<Packet>;
    ser.setCheckfuc(check_fuc);
    int ret = ser.openDevice("/dev/ttyACM0", 460800);

    if(ret == 1)
        std::cout<<"serial open ok"<<"\n";
    else
        std::cerr<<"serial open err: "<<ret<<"\n";

    ser.startReceive(100);


    //2.0初始化相机
    Hik.continueCap(3);

    //3.0创建数据配对线程，并将数据发布到Frames环形队列
    std::thread match_thread = std::thread(rm::IMUAndImageMatchFunction, std::ref(Hik), std::ref(ser), std::ref(Frames));
    std::thread plan_thread = std::thread(rm::MPCPlanFunction, std::ref(planner), std::ref(robotStates), std::ref(ser));

    // cv::namedWindow("frame");
        auto start = std::chrono::steady_clock::now();
    std::printf("Start main loop\n");

    while(true)
    {
        if(Frames.empty()) continue;

        FrameData frame;

        while (!Frames.empty())
        {
            Frames.pop(frame);
        }
        if(frame.image.empty()) {std::this_thread::sleep_for(std::chrono::microseconds(100));continue;}

        // cv::imshow("frame", frame.image);
        // cv::waitKey(1);
        
                // 性能统计
        // test.count(std::chrono::steady_clock::now() - start);
        // start = std::chrono::steady_clock::now();

        // if(test.num%200 == 0 && test.num != 0) {
        //     test.show();
        //     test.clear();
        // }
        //计算枪管方向

        const Eigen::Matrix<double, 3, 1>& Gun = shoot.GunDirection(Eigen::Quaterniond{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z});

    //   auto t1_ = std::chrono::steady_clock::now();
        // 1. 传统视觉检测
        std::vector<cv::Mat> armors_pattern;
        
        auto opencv_armors = detect(frame.image, armors_pattern);

        // std::cout<<"opencv_armors num:" << opencv_armors.size() << "\n";
        Eigen::Quaterniond gripper_to_world{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z};
        std::vector< std::array<ArmorPosi,2> > armors_2 = Sov(opencv_armors, gripper_to_world);


        // 2. 数字分类
        std::vector<ArmorPosi> armors = resnet(armors_2, armors_pattern);

        //结果
        // for(auto& armor : armors)
        // {
        //     Sov.ansShow(armor.posi,frame.image);
        // }


        // cv::imshow("frame", frame.image);
        // cv::waitKey(1);

        // std::cout<<"yolo_armors num:" << yolo_armors.size() << "\n";


        // std::cout<<"fused_yolo_armors num:" << fused_yolo_armors.size() << "\n";

        // std::cout<<"time: " << (std::chrono::steady_clock::now()-t1_).count() <<"\n";
        // std::cout<<"FilterAndConverToWorld armors_posi num:" << armors_posi.size() << "\n";

        // 7. 使用Tracker进行追踪
        double dt = rm::SolveDt(next_point, frame.time, 0.005);
        track(armors, frame.quat, Gun, dt);
        next_point = frame.time;

        // 8. 获取当前追踪的机器人
        const auto& current_robot = track.getCurrentRobot();

        if (current_robot == nullptr)
        {
            // 没有追踪到目标，跳过
            robotStates.push(nullptr);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            rm::SendMessageToRobot(ser, 0.0, 0.0, false);
            continue;
        }

        if(current_robot->GetMode() == Robot::KalmanMode::EKF)
        {
            robotStates.push(std::make_unique<RobotState>(*current_robot,frame.time));
        }
        else {
            robotStates.push(nullptr);
            double dt = shoot.FlyTime(current_robot->center);
            Eigen::Vector3d aim = current_robot->center + current_robot->Speed.block<3,1>(0,0) * dt;
             
            std::array<double, 2> Pitch_and_Yaw = shoot(aim);
            rm::SendMessageToRobot(ser, Pitch_and_Yaw[0], Pitch_and_Yaw[1], true);
        }
        

        // 性能统计
        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();

        if(test.num%200 == 0 && test.num != 0) {
            test.show();
            test.clear();
        }
        #ifdef MainDebug
            viz.update(*current_robot, current_robot->Predict(0), dt, Gun);
        #endif
        // 可视化
        // for(const auto& armor_posi : armors_posi)
        // {
        //     Sov.ansShow(armor_posi.posi, frame.image);
        // }
        // cv::imshow("frame", frame.image);
        // cv::waitKey(1);
    }


    return 0;
}




//性能测试工具
void Test::count(const std::chrono::nanoseconds& time)
{
    this->num++;
    total += time;
}

void Test::clear()
{
    this->num = 0;
    this->total = std::chrono::nanoseconds(0);
}

void Test::show()
{
    std::cout<< this->num / ( ( (double)this->total.count() ) * 1e-9 )<< "Hz\n" ;
}

