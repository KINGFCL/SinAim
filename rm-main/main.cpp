#include "include/HikCamera.hpp"
#include "include/RTSerial.hpp"
#include "include/fastqueue.hpp"
#include "include/Detector.hpp"
#include "include/Solver.hpp"
#include "include/Shooter.hpp"
#include "include/Target.hpp"
// #include "../../rm-main/include/Tracker.hpp"
#include "include/ShootTable.hpp"
#include "include/Data.hpp"
#include "include/IMUAndImageMatch.hpp"

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

io::HikCamera Hik(1,17);
io::RTSerial<Packet> ser(20);

Detector detect(Light::Color::Blue,0.5,"../../../rm-main/model/mobilenet_v3_112_rgb.onnx");
Solver Sov("../../../config/Solver_config.yaml");
Robot robot;
// Tracker track;


ShootTable::TableConfig tableconfig(10,0,2,-1,0.01,"/home/king/AUTO-Aming-system/config/infantry_10_table.bin");
Shooter shoot(cv::Point3d(-0.9996123276310385,0.02082249458349189, -0.01848291555403893),tableconfig);

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
    Hik.continueCap(5);

    //3.0创建数据配对线程，并将数据发布到Frames环形队列
    std::thread match_thread = std::thread(IMUAndImageMatchThread, std::ref(Hik), std::ref(ser), std::ref(Frames));

    cv::namedWindow("frame");
    auto start = std::chrono::steady_clock::now();

    while(true)
    {
        FrameData frame;
        bool haveData = Frames.pop(frame);
        
        if(!haveData) continue;
        
        //如果不是最新照片直接跳过直到拿到最新照片
        if(!Frames.empty()) continue;

        auto armors = detect(frame.image);

        //解算装甲板位置
        auto armors_posi = Sov(armors);

        Robot::SolveRobotSize(armors_posi);


        // #endif
        // if(test.num%100 == 0 && test.num != 0)
        // {
        //     std::cout<<armors_posi[0].posi<<"\n";
        // }
        for(const auto& armor_posi : armors_posi)
        {
            Sov.ansShow(armor_posi.posi,frame.image);
        }
        detect.ArmorShow(frame.image, armors);
        cv::imshow("frame", frame.image);
        
        cv::waitKey(1);
        Sov.ConverToWorld(armors_posi,frame.quat);


        


        // if(test.num%100 == 0 && test.num != 0)
        // {
        //     std::cout<<armors_posi[0].posi/10<<"\n";
        // }
        // std::cout<<"quat: "<<frame.quat.w<<" "<<frame.quat.x<<" "<<frame.quat.y<<" "<<frame.quat.z<<"\n";

        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();

        if(test.num%200 == 0 && test.num != 0) {test.show();
            test.clear();}


        //打弹


        // std::this_thread::sleep_for(std::chrono::nanoseconds(100000000));

        //traker:

        // Eigen::Matrix<double, 3, 1> posi;
        // posi << armors_posi[0].posi.x, armors_posi[0].posi.y, armors_posi[0].posi.z;

        // auto ans = track(posi,0.004);
        // // std::cout<< "Filtered Position: " << ans.transpose() << std::endl;

        // float dt = shoot.FlyTime(armors_posi[0].posi/1000); 

        // cv::Point3d predict_posi;
        // predict_posi.x = (ans(0,0) + dt * ans(3,0)) ;
        // predict_posi.y = (ans(1,0) + dt * ans(4,0)) ;
        // predict_posi.z = (ans(2,0) + dt * ans(5,0)) ;
        Two_Robot.Update(armors_posi);

        auto predict_posi = armors_posi[0].posi/1000;//单位换算到m
        // std::cout<< "Predict Position: " << predict_posi << "\n";

        std::array<double, 2> Pitch_and_Yaw = shoot(predict_posi);
        // std::cout<<Pitch_and_Yaw[0]<<" "<<Pitch_and_Yaw[1]<<"\n";
        ShootPosi sed1;
        sed1.row =0 ;
        sed1.pitch = (float)Pitch_and_Yaw[0];
        // std::cout<<sed1.pitch<<"\n";
        sed1.yaw = (float)Pitch_and_Yaw[1];
        // std::cout<<sed1.pitch<<"  "<<sed1.yaw<<"\n";
        sed1.checksum = io::CRC8::Calculate(&sed1, sizeof(sed1)-1);

        ShootFire sed2;
        sed2.fire = 1;
        sed2.checksum = io::CRC8::Calculate(&sed2, sizeof(sed2)-1);

        ser.writeBytes(&sed1,sizeof(sed1));
        ser.writeBytes(&sed2,sizeof(sed2));
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

