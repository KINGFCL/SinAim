#include "include/HikCamera.hpp"
#include "include/NumClassifier.hpp"
#include "include/RTSerial.hpp"
#include "include/fastqueue.hpp"
#include "include/Detector.hpp"
#include "include/Solver.hpp"
#include "include/Shooter.hpp"
#include "include/Target.hpp"
// #include "../../rm-main/include/Tracker.hpp"
#include "include/ShootTable.hpp"
#include "include/Data.hpp"
#include "Function.hpp"

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

Detector detect(Light::Color::Blue,0.5);//,"../../../rm-main/model/mobilenet_v3_112_rgb.onnx"
NumClassifier classifier("../../../rm-main/model/mobilenet_v3_112_rgb.onnx","../../../config/NumClassifier_config.yaml");

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
    std::thread match_thread = std::thread(rm::IMUAndImageMatchFunction, std::ref(Hik), std::ref(ser), std::ref(Frames));

    cv::namedWindow("frame");
    auto start = std::chrono::steady_clock::now();

    while(true)
    {

        if(Frames.empty()) continue;

        FrameData frame;
        
        while (!Frames.empty()) 
        {
            Frames.pop(frame);
        }
        std::vector<cv::Mat> armors_pattern;

        auto armors = detect(frame.image,armors_pattern);

        //解算装甲板位置
        auto armors_posis = Sov(armors);

        Sov.Filter(armors_posis, armors_pattern);

        auto armors_posi = classifier(armors_posis,armors_pattern);

        Robot::SolveRobotSize(armors_posi);

        // for(const auto& armor_posi : armors_posi)
        // {
        //     Sov.ansShow(armor_posi.posi,frame.image);
        // }

        detect.ArmorShow(frame.image, armors);
        cv::imshow("frame", frame.image);
        
        cv::waitKey(1);

        Sov.ConverToWorld(armors_posi,frame.quat);
        robot.Update(armors_posi,0.005);

        auto aim = robot.Predict(dt);
        // std::cout<<"aim: "<<aim<<"\n";


        


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

