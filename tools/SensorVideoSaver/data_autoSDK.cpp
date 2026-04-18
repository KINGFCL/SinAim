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
    
    cv::namedWindow("hhh");

    auto start = std::chrono::steady_clock::now();
    
    // 1. 申请一块巨大的连续内存作为 RAM 缓冲区
    // 预先分配空间可以防止 vector 运行中途扩容导致的卡顿
    std::vector<FrameData> ram_buffer;
    ram_buffer.reserve(2000); // 预留 2000 帧的空间 (约 9.2 GB 内存，视你需要调整)

    std::cout << "\n>>> 正在录制至内存... 按 ESC 结束并写入磁盘 <<<" << std::endl;

    while(true)
    {
        FrameData frame;
        bool haveData = Frames.pop(frame);
        
        if(!haveData) continue;
        
        // 激进的丢帧逻辑可以保留，确保你拿到的是最新帧
        if(!Frames.empty()) continue;

        // 2. 将数据推入 RAM 缓冲区 (必须 clone 图像！)
        FrameData buffered_frame;
        buffered_frame.image = frame.image.clone(); // 极其重要：深拷贝图像内存！
        buffered_frame.quat = frame.quat;
        buffered_frame.time = frame.time;
        
        ram_buffer.push_back(buffered_frame);

        // 性能统计
        test.count(std::chrono::steady_clock::now() - start);
        start = std::chrono::steady_clock::now();

        if(test.num % 200 == 0 && test.num != 0) {
            test.show();
            test.clear();
        }

        // 显示图像与退出检测
        cv::imshow("hhh", frame.image);
        int key = cv::waitKey(1);
        if (key == 27) {
            break; // 收到 ESC 信号，跳出高频采集循环
        }
    }

    // ==========================================
    // 后处理：将 RAM 缓冲区的数据集中写入 SSD
    // ==========================================
    std::cout << "\n[系统] 停止录像，开始将 RAM 中的 " << ram_buffer.size() << " 帧数据进行无损编码并写入磁盘..." << std::endl;
    std::cout << "[系统] 这个过程可能需要一些时间，请勿关闭程序！" << std::endl;

    if (!ram_buffer.empty()) {
        auto last_point = ram_buffer[0].time;
        
        for (size_t i = 0; i < ram_buffer.size(); ++i) {
            auto quat = ram_buffer[i].quat;
            double dt = SolveDt(last_point, ram_buffer[i].time, 0.01);
            
            // 写入视频 (这一步此时会霸占 CPU 进行 FFV1 压缩)
            saver.save(ram_buffer[i].image, Eigen::Quaterniond(quat.w, quat.x, quat.y, quat.z), dt);
            last_point = ram_buffer[i].time;

            // 打印进度条
            if (i % 20 == 0) {
                std::cout << "编码写入进度: " << i << " / " << ram_buffer.size() << " 帧\r" << std::flush;
            }
        }
    }

    std::cout << "\n[系统] 视频保存完毕！安全退出。" << std::endl;

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

