#ifndef INCLUDE_LIBXR_ROBOT_LINK_HPP
#define INCLUDE_LIBXR_ROBOT_LINK_HPP

#include <chrono>
#include <cstddef>
#include <memory>

#include "opencv2/core/quaternion.hpp"

namespace io
{

class LibXRRobotLink
{
public:
    using timePoint = std::chrono::steady_clock::time_point;

    explicit LibXRRobotLink(std::size_t quat_buffer_size = 256);
    ~LibXRRobotLink();

    LibXRRobotLink(const LibXRRobotLink&) = delete;
    LibXRRobotLink& operator=(const LibXRRobotLink&) = delete;

    bool start(const char* vid = "16d0",
               const char* pid = "1492",
               unsigned int baudrate = 115200);

    bool readQuat(cv::Quatd& quat, timePoint& data_time);

    void publishCommand(float pitch, float yaw, bool fire);

private:
    struct Impl;
    std::size_t quat_buffer_size_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace io

#endif
