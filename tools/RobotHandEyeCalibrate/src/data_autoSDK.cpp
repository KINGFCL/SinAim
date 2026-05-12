#include "HikCamera.hpp"
#include "LibXRSerial.hpp"
#include "libxr.hpp"
#include "linux_uart.hpp"

#include <Eigen/Geometry>
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

namespace
{
constexpr const char* kSerialDevice = "/dev/ttyACM0";
constexpr unsigned int kSerialBaud = 460800;
}

struct FrameData
{
    cv::Mat image;
    Eigen::Quaterniond gripper_to_world;
    std::chrono::steady_clock::time_point time;

    FrameData(const cv::Mat image, const Eigen::Quaterniond& gripper_to_world,
              const std::chrono::steady_clock::time_point& time)
        : image(image), gripper_to_world(gripper_to_world), time(time) {}
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
template <std::size_t BufferSize>
void IMUAndImageMatchThread(io::HikCamera& Hik, io::LibXRSerial<BufferSize>& ser,FastQueue<FrameData>& Frames);


io::HikCamera Hik(2,10);
static FastQueue<FrameData> Frames(10);


Test test;


// 函数：生成一个基于当前时间的唯一文件名
static int Num=0;

std::string generate_filename(int &Num) {
    
    std::string ss;
    ss ="image_"+std::to_string(Num++)+"_";
    return ss;
}

int main() {
    LibXR::PlatformInit();
    LibXR::RamFS ramfs;
    LibXR::LinuxUART uart(kSerialDevice, kSerialBaud);
    LibXR::HardwareContainer hw(
        LibXR::Entry<LibXR::LinuxUART>{uart, {"DevC-USB"}},
        LibXR::Entry<LibXR::RamFS>{ramfs, {"ramfs"}});
    LibXR::ApplicationManager appmgr;
    io::LibXRSerial<> ser(hw, appmgr);

    // 1. 定义数据保存目录
    std::string output_dir = "../Data/images";
    std::string config_path = "../Data/Calibration_R_T.yaml";

    //加载存储数据的YAML文件
    cv::FileStorage fs;
    if (!fs.open(config_path, cv::FileStorage::WRITE)) {
        std::cerr << "Error: Failed to open YAML file: " << config_path << std::endl;
        return -1;  // 失败时返回
    }


    std::cout << "Successfully opened " << config_path << std::endl;
    std::cout << "------------------------------------------" << std::endl;





    //1.0初始化串口
    std::cout<<"serial init ok: "<<kSerialDevice<<" @ "<<kSerialBaud<<"\n";


    //2.0初始化相机
    Hik.continueCap(5);

    //3.0创建数据配对线程，并将数据发布到Frames环形队列
    std::thread match_thread([&]() { IMUAndImageMatchThread(Hik, ser, Frames); });
    
    cv::namedWindow("frame");

    auto start = std::chrono::steady_clock::now();
    while(true)
    {
        FrameData frame;
        bool haveData = Frames.pop(frame);
        
        if(!haveData) continue;
        
        //如果不是最新照片直接跳过直到拿到最新照片
        if(!Frames.empty()) continue;

        cv::imshow("frame", frame.image);
        
        // 等待按键事件 (等待1毫秒)
        // 这个延时对于显示视频至关重要，否则窗口会无响应
        int key = cv::waitKey(1);

        // 9. 处理按键
        if (key == ' ') { // 空格键的ASCII码是32
            // 生成文件名并拼接完整路径
            std::string filename = generate_filename(Num);
            std::string filepath = output_dir + "/" + filename;

            // 保存当前帧为PNG图片
            bool saved = cv::imwrite(filepath+".png", frame.image);
            const Eigen::Matrix3d R_grip_to_world_eigen =
                frame.gripper_to_world.toRotationMatrix();
            cv::Mat R_grip_to_world(3, 3, CV_64F);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    R_grip_to_world.at<double>(row, col) =
                        R_grip_to_world_eigen(row, col);
                }
            }
            cv::Mat R_world_to_grip = R_grip_to_world.t();

            fs << filename << R_world_to_grip;
            std::cout<<R_world_to_grip<<"\n";

            
            if (saved) {
                std::cout << "图片已保存: " << filepath << std::endl;
            } else {
                std::cerr << "错误: 无法保存图片到 " << filepath << std::endl;
            }
        } else if (key == 27) { // ESC键的ASCII码是27
            std::cout << "正在退出程序..." << std::endl;
            break; // 退出循环
        }


        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();

        if(test.num%200 == 0 && test.num != 0) {test.show();test.clear();}
    }

    fs.release();// 关闭文件
    cv::destroyAllWindows();
    match_thread.detach();
    return 0;
}


//IMU与图像配对线程
template <std::size_t BufferSize>
void IMUAndImageMatchThread(io::HikCamera& Hik, io::LibXRSerial<BufferSize>& ser,FastQueue<FrameData>& Frames)
{
    while (true) {

        // 读取相机数据
        io::HikCamera::ImageData HikData;

        Hik.read(HikData);
        if(HikData.image.empty()) continue;

        // 读取串口数据
        std::chrono::steady_clock::time_point time ;
        Eigen::Quaterniond gripper_to_world;
        while( true )
        {
            bool ret = ser.ReadData(gripper_to_world, time);
            if( !ret ) break;

            const auto& t = ((double)(HikData.time - time).count()) * 1e-6;

            //配对超时

            //串口数据比相机数据早8ms以上
            if( t > 8 ) continue;

            //串口数据比相机数据早5ms以下
            if( t < 5 ) break;

            //配对成功
            FrameData frame(HikData.image, gripper_to_world, HikData.time);

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
