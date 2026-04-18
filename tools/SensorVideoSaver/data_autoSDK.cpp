#include "HikCamera.hpp"
#include "fastqueue.hpp"
#include "RTSerial.hpp"
#include "SensorVideoSaver.hpp"
#include <opencv2/core/quaternion.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/core/matx.hpp>
#include <string>
#include <iomanip>    // 用于格式化时间字符串
#include <sstream>    // 用于构建字符串
#include <thread>

//空格键保存图像，ESC键退出程序

struct __attribute__((packed)) Packet{

    uint8_t header;       // 0xA5
    uint8_t target_id;    // 0xEA
    uint8_t length;       // 0x1A (26)
    uint8_t cmd_id;       // 0x35
    uint8_t head_chk;     // 0xA6
    uint32_t timestamp;   // 0x7A100000 (Little Endian) or ID
    float q0;             // x
    float q1;             // y
    float q2;             // z
    float q3;             // w
    uint8_t checksum;     // 校验和
} ;

struct FrameData
{
    cv::Mat image;
    cv::Quatd quat;
    std::chrono::steady_clock::time_point time;

    FrameData(const cv::Mat image, const cv::Quatd& quat,
              const std::chrono::steady_clock::time_point& time)
        : image(image), quat(quat), time(time) {}
    FrameData(){}
};

//性能测试工具
struct Test
{
    int num = 0;
    std::chrono::nanoseconds total{0};

    void count(const std::chrono::nanoseconds& time);
    void clear();
    void show();
};

//IMU与图像配对线程
void IMUAndImageMatchThread(io::HikCamera& Hik, io::RTSerial<Packet>& ser,FastQueue<FrameData>& Frames);

double SolveDt(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end, double pic);
io::HikCamera Hik(2,10);
io::RTSerial<Packet> ser(20);
static FastQueue<FrameData> Frames(10);


Test test;


std::string output_dir = "../Data/damo.avi";
SensorVideoSaver saver(output_dir);



int main() {
    // 1. 定义数据保存目录



    //加载存储数据的YAML文件


    //1.0初始化串口
    std::cout<<sizeof(Packet)<<std::endl;

    std::function<bool(const Packet&)> check_fuc = io::CRC8::Check<Packet>;
    ser.setCheckfuc(check_fuc);
    int ret = ser.openDevice("/dev/ttyACM1", 460800);
    
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
    auto last_point = std::chrono::steady_clock::now();

    while(true)
    {
        FrameData frame;
        bool haveData = Frames.pop(frame);
        
        if(!haveData) continue;
        
        //如果不是最新照片直接跳过直到拿到最新照片
        if(!Frames.empty()) continue;

        auto quat = frame.quat;
        saver.save(frame.image, Eigen::Quaterniond(quat.w, quat.x, quat.y, quat.z), SolveDt(last_point,frame.time,0.01));
        last_point = frame.time;

        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();

        if(test.num%200 == 0 && test.num != 0) {test.show();test.clear();}
    }

    match_thread.detach();
    return 0;
}


//IMU与图像配对线程
void IMUAndImageMatchThread(io::HikCamera& Hik, io::RTSerial<Packet>& ser,FastQueue<FrameData>& Frames)
{
    while (true) {

        // 读取相机数据
        io::HikCamera::ImageData HikData;

        Hik.read(HikData);
        if(HikData.image.empty()) continue;

        // 读取串口数据
        std::chrono::steady_clock::time_point time ;
        Packet IMU;
        while( true )
        {
            bool ret = ser.readPacket(IMU, time);
            if( !ret ) break;

            const auto& t = ((double)(HikData.time - time).count()) * 1e-6;

            //配对超时

            //串口数据比相机数据早8ms以上
            if( t > 8 ) continue;

            //串口数据比相机数据早5ms以下
            if( t < 5 ) break;

            //配对成功
            cv::Quatd quat( IMU.q3, IMU.q0, IMU.q1, IMU.q2 );
            FrameData frame(HikData.image, quat, HikData.time);

            Frames.push(frame);
            break;
        }

    }
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

double SolveDt(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end, double pic) {
    // 1. 安全检查：避免除以零
    if (std::abs(pic) < 1e-9) {
        return 0.0;
    }

    // 2. 获取时间差，并将单位转换为秒 (s)
    // std::chrono::duration<double> 默认就等同于 std::chrono::duration<double, std::ratio<1>>，也就是秒。
    std::chrono::duration<double> diff = end - start;
    double dt = diff.count();

    // 3. 寻找最接近的倍数 n
    double n = std::round(dt / pic);

    // 4. 保证 n 为非零整数
    if (n == 0.0) {
        n = (dt >= 0) ? 1.0 : -1.0;
    }

    // 5. 返回最接近的 n * pic (单位：秒)
    return n * pic;
}

