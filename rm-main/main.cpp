#include "Function.hpp"
#include "communicate/RerunVisualizer.hpp"


#include <chrono>
#include <cstdio>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <thread>

#define SmallMainDebug
#ifdef SmallMainDebug
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

std::chrono::steady_clock::time_point next_point = std::chrono::steady_clock::now();

io::HikCamera Hik(1,16);
io::RTSerial<Packet> ser(80);

// 传统视觉检测器
CVDetector detect(Light::Color::Blue);

// YOLO检测器
YOLODetector yolo11detect("../model/yolo11.xml", YOLODetector::Camp::Blue);

RerunVisualizer viz("RoboMaster_AutoAim");

Solver::SolverConfig solver_config{
    {/* camera_matrix 3x3, 按行填写 */
     1000.0, 0.0, 640.0,
     0.0, 1000.0, 360.0,
     0.0, 0.0, 1.0},
    {/* distortion_coeffs k1,k2,p1,p2,k3 */
     0.0, 0.0, 0.0, 0.0, 0.0},
    {/* R_Cam_to_gripper 3x3, 按行填写 */
     1.0, 0.0, 0.0,
     0.0, 1.0, 0.0,
     0.0, 0.0, 1.0},
    {/* T_Cam_to_gripper x,y,z (cm) */
     0.0, 0.0, 0.0},
    1.0 /* reproj_threshold */
};
Solver Sov(solver_config);

// 追踪器
Tracker track;

Shooter shoot(Eigen::Matrix<double,3,1>(0.9999283656297303,
 0.002822337907989043,
 -0.01163176761242803));

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
        if(frame.image.empty()) continue;

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
        auto opencv_armors_deque = detect(frame.image, armors_pattern);

        // std::cout<<"opencv_armors num:" << opencv_armors_deque.size() << "\n";

        // 2. YOLO检测
        std::vector<YoloArmor> yolo_armors = yolo11detect(frame.image);

        // std::cout<<"yolo_armors num:" << yolo_armors.size() << "\n";

        // 3. 融合传统视觉和YOLO的结果
        auto fused_yolo_armors = rm::MatchYoloAndOpenCV(opencv_armors_deque, yolo_armors);

        // yolo11detect.draw(frame.image, fused_yolo_armors);
        // cv::imshow("frame", frame.image);
        // cv::waitKey(1);
        // std::cout<<"fused_yolo_armors num:" << fused_yolo_armors.size() << "\n";

        // 4. 将融合结果转为 CVArmor 传给 Solver
        Eigen::Quaterniond gripper_to_world{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z};
        std::vector<CVArmor> cv_armors;
        for (const auto& ya : fused_yolo_armors) {
            if (ya.keypoints.size() < 4) continue;
            Light l_left(cv::RotatedRect(ya.keypoints[0], ya.keypoints[3], ya.keypoints[0]));
            Light l_right(cv::RotatedRect(ya.keypoints[1], ya.keypoints[2], ya.keypoints[1]));
            CVArmor ca(l_left, l_right);
            ca.Lightcorners = ya.keypoints;
            cv_armors.push_back(ca);
        }

        // 5. 解算装甲板位置，内部完成坐标系转换
        auto armors_2 = Sov(cv_armors, gripper_to_world);

        // 取每对中重投影误差更小的一侧
        std::vector<ArmorPosi> armors_posi;
        for (const auto& pair : armors_2) {
            armors_posi.push_back(pair[0].reproj < pair[1].reproj ? pair[0] : pair[1]);
        }
        // std::cout<<"time: " << (std::chrono::steady_clock::now()-t1_).count() <<"\n";
        // std::cout<<"armors_posi num:" << armors_posi.size() << "\n";
        
        // 7. 使用Tracker进行追踪
        track(armors_posi, frame.quat, Gun, rm::SolveDt(next_point, frame.time, 0.01));
        next_point = frame.time;

        // 8. 获取当前追踪的机器人
        Robot* current_robot = track.getCurrentRobot();

        if (current_robot == nullptr)
        {
            // 没有追踪到目标，跳过
            continue;
        }

        // 9. 预测和瞄准
        double dt = shoot.FlyTime(current_robot->center);
        Eigen::Matrix<double, 4, 4> aims = current_robot->Predict(dt);

        Eigen::Matrix<double, 4, 1> aim = rm::ChooseBestAimArmor(aims, current_robot->Speed, Gun);

        std::cout<<aim<<"\n";
        // std::cout<<aim.norm()<<"\n";
        // 10. 计算射击角度
        std::array<double, 2> Pitch_and_Yaw = shoot(aim.block<3,1>(0,0));

        // 11. 发送控制指令
        bool fire_ = rm::CheckFireCondition(frame.quat, Pitch_and_Yaw, aim, Gun, 0.03, 0.03);
        rm::SendMessageToRobot(ser, Pitch_and_Yaw[0], Pitch_and_Yaw[1], true);
        // std::cout<<fire_<<"\n";
        // std::cout<<"Pitch: "<<Pitch_and_Yaw[0]-0.05<<" Yaw: "<<Pitch_and_Yaw[1]<<"\n";
        // 性能统计
        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();

        if(test.num%200 == 0 && test.num != 0) {
            test.show();
            test.clear();
        }
        #ifdef SmallMainDebug
            viz.update(*current_robot, aims, dt, Gun);
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