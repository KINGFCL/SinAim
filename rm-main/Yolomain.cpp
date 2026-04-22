#include "Function.hpp"
#include "communicate/RerunVisualizer.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <thread>

#define MainDebug
#ifdef MainDebug
double R_sum = 0.0;
int R_count = 0;
#endif
//Debug


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

io::HikCamera Hik(1.5,15);
io::RTSerial<Packet> ser(20);


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
Robot robot;
// Tracker track;

Shooter shoot(Eigen::Matrix<double,3,1>(-0.9972026403208731, 0.001749666619733665, -0.07472504803477144));

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

    cv::namedWindow("frame");
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
        //计算枪管方向
        Eigen::Quaterniond gripper_to_world{frame.quat.w, frame.quat.x, frame.quat.y, frame.quat.z};
        const Eigen::Matrix<double, 3, 1> Gun = shoot.GunDirection(gripper_to_world);


        auto yolo_armors = yolo11detect(frame.image);
        // yolo11detect.draw(frame.image, yolo_armors);
        // cv::imshow("frame",frame.image);
        // cv::waitKey(1);

        // std::cout<<"------------------------------------------------\n";

        // std::cout<<"detect num: "<<yolo_armors.size()<<"\n";

        // 将 YoloArmor keypoints 转为 CVArmor 以复用 Solver
        std::vector<CVArmor> cv_armors;
        for (const auto& ya : yolo_armors) {
            if (ya.keypoints.size() < 4) continue;
            Light l_left(cv::RotatedRect(ya.keypoints[0], ya.keypoints[3], ya.keypoints[0]));
            Light l_right(cv::RotatedRect(ya.keypoints[1], ya.keypoints[2], ya.keypoints[1]));
            CVArmor ca(l_left, l_right);
            ca.Lightcorners = ya.keypoints;
            cv_armors.push_back(ca);
        }

        //解算装甲板位置
        auto armors_2 = Sov(cv_armors, gripper_to_world);

        // std::cout<<"after filter num: "<<armors_2.size()<<"\n";

        // 取每对中重投影误差更小的一侧作为 armors_posi
        std::vector<ArmorPosi> armors_posi;
        for (const auto& pair : armors_2) {
            armors_posi.push_back(pair[0].reproj[0] < pair[1].reproj[0] ? pair[0] : pair[1]);
        }

        // for(const auto& armor_posi : armors_posi)
        // {
        //     Sov.ansShow(armor_posi.posi,frame.image);
        // }

        // cv::imshow("frame", frame.image);

        // cv::waitKey(1);

        if(armors_posi.empty()) {robot.Update(rm::SolveDt(next_point, frame.time, 0.005));}
        else { robot.Update(armors_posi, rm::SolveDt(next_point, frame.time, 0.005)); }
        next_point = frame.time;

        // #ifdef MainDebug
        // std::cout<<"------------------------------------------------\n";
        // std::cout<<"armors_posi: "<< "\n" <<armors_posi[0].posi<<"\n";
        // #endif


        double dt = shoot.FlyTime(robot.center);
        auto aims = robot.Predict(dt);

        #ifdef MainDebug
        if(test.num%4 == 0 && test.num != 0)
            viz.update(robot, aims, dt,  Gun);
        #endif

        auto aim =  rm::ChooseBestAimArmor(aims, robot.Speed, Gun);
        
        // if(test.num%100 == 0 && test.num != 0)
        // {
        //     std::cout<<armors_posi[0].posi/10<<"\n";
        // }
        // std::cout<<"quat: "<<frame.quat.w<<" "<<frame.quat.x<<" "<<frame.quat.y<<" "<<frame.quat.z<<"\n";

        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();

        if(test.num%200 == 0 && test.num != 0) {test.show();test.clear();}


        //打弹
        
        // std::cout<< "aim: " << aim << "\n";

        // Shooter::operator() 接受 Eigen::Matrix<double,3,1>，单位 cm
        Eigen::Matrix<double, 3, 1> predict_posi = aim.block<3,1>(0,0);
        // std::cout<< "Predict Position: " << predict_posi.transpose() << "\n";

        std::array<double, 2> Pitch_and_Yaw = shoot(predict_posi);
        // if(0.1 < std::abs(Pitch_and_Yaw[1])  || std::abs(Pitch_and_Yaw[1]) < 0.2 )
        // {
        //     std::cerr<<"aim error: "<< Pitch_and_Yaw[0]<<" "<<Pitch_and_Yaw[1]<<"\n"<< robot.Speed<<"\n";
            
        // }else {
        //     std::cout<<"aim ok: "<< Pitch_and_Yaw[0]<<" "<<Pitch_and_Yaw[1]<<"\n"<< robot.Speed<<"\n";
        // }

        // std::cout<<Pitch_and_Yaw[0]<<" "<<Pitch_and_Yaw[1]<<"\n";
        // rm::SendMessageToRobot(ser, Pitch_and_Yaw[0], Pitch_and_Yaw[1] , true);
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

