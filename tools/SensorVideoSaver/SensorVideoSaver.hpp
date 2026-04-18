#include <opencv2/opencv.hpp>
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include <iostream>
#include <string>
#include <cstring>

class SensorVideoSaver {
private:
    cv::VideoWriter writer_;
    std::string filename_;
    double fps_;
    bool is_initialized_;

    // 改为 double 类型的结构体，共 32 bytes
    #pragma pack(push, 1)
    struct DoubleQuaternion {
        double time; // 这张照片和上一个照片的时间差
        double w;
        double x;
        double y;
        double z;
    };
    #pragma pack(pop)

public:
    explicit SensorVideoSaver(const std::string& filename = "damo.avi", double fps = 100.0) 
        : filename_(filename), fps_(fps), is_initialized_(false) {}

    ~SensorVideoSaver() {
        if (writer_.isOpened()) {
            writer_.release();
        }
    }

    SensorVideoSaver(const SensorVideoSaver&) = delete;
    SensorVideoSaver& operator=(const SensorVideoSaver&) = delete;

    // 唯一接口
    void save(cv::Mat& image, const Eigen::Quaterniond& quat, double time) {
        if (image.empty()) {
            std::cerr << "[SensorVideoSaver] Warning: 收到空图像，已忽略。" << std::endl;
            return;
        }

        // 1. 延迟初始化 VideoWriter
        if (!is_initialized_) {
            // 必须保留无损编码 FFV1，double 数据被压缩会变成乱码
            int fourcc = cv::VideoWriter::fourcc('F', 'F', 'V', '1'); 
            writer_.open(filename_, fourcc, fps_, cv::Size(image.cols, image.rows));
            
            if (!writer_.isOpened()) {
                std::cerr << "[SensorVideoSaver] Error: 无法打开文件 " << filename_ << std::endl;
                return;
            }
            is_initialized_ = true;
            std::cout << "[SensorVideoSaver] 录制已启动 (" << image.cols << "x" << image.rows << ") - Double 模式" << std::endl;
        }

        // 2. 重新排列为 [w, x, y, z] 的顺序 (Eigen 内部默认是 [x, y, z, w])
        DoubleQuaternion dq;
        dq.time = time;
        dq.w = quat.w();
        dq.x = quat.x();
        dq.y = quat.y();
        dq.z = quat.z();

        // 3. 内存拷贝 (32 bytes)
        if (image.total() * image.elemSize() >= sizeof(DoubleQuaternion)) {
            std::memcpy(image.data, &dq, sizeof(DoubleQuaternion));
        } else {
            std::cerr << "[SensorVideoSaver] Error: 图像太小，无法写入 32 字节的四元数！" << std::endl;
            return; 
        }

        // 4. 存入视频包
        writer_.write(image);
    }
};