#ifndef IO_HIKCAMERA_HPP
#define IO_HIKCAMERA_HPP

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <condition_variable>
#include <chrono>

#include "FastQueue.hpp"
#include "MvCameraControl.h"
#include "opencv2/opencv.hpp"
#include "thread_safe_queue.hpp"

namespace io{
using namespace std::chrono_literals;

class HikCamera
{
public:
    
    struct ImageData
    {
        cv::Mat image;
        std::chrono::steady_clock::time_point time;
    };

    struct HikCameraConfig
    {
        double exposure_ms;
        double gain;
        bool   autocap = true;
        bool   IsFlip = false;
    };

    HikCamera(double exposure_ms,
              double gain,
              bool autocap=true,
              bool IsFlip = false);
    explicit HikCamera(const HikCameraConfig& config);

    void read(ImageData& imgdata);
    void continueCap(size_t MaxframeNum);

    ~HikCamera();

private:

    enum class Hik {Running,Stopped};
    struct config
    {
        double exposure_ms;
        double gain;
        bool autocap;
        bool IsFlip = false;
    };
    struct protect
    {

        std::mutex mux;
        std::condition_variable HikIsquit;
        std::thread protectthread;    
    };
    
    
    int vid_, pid_;
    void * handle_;
    
    config parame;
    protect guard;

    bool conCapOpen = false;
    
    std::atomic<Hik> HikState;
    std::thread HikSDKthread;
    size_t MaxframeNum=0;
    std::unique_ptr<FastQueue<ImageData>> Frames_ptr;

    void ProtectRunning();

    void capture_init();

    void capture_stop();

    void set_float_value(const std::string & name, double value);
    void set_enum_value(const std::string & name, unsigned int value);

};

}
#endif 










