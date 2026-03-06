#ifndef INCLUDE_FUNCTION_HPP
#define INCLUDE_FUNCTION_HPP

#include "include/Armor.hpp"
#include "include/Data.hpp"
#include "include/HikCamera.hpp"
#include "include/RTSerial.hpp"
#include "include/fastqueue.hpp"

#include <deque>
#include <eigen3/Eigen/Core>
#include <opencv2/core/types.hpp>

//IMU与图像配对线程逻辑
namespace rm
{
    void IMUAndImageMatchFunction(io::HikCamera& Hik, io::RTSerial<Packet>& ser,FastQueue<FrameData>& Frames);
    void SendMessageToRobot(io::RTSerial<Packet>& ser, float pitch, float yaw, bool fire);
    Eigen::Matrix<double, 3, 1> ChooseBestAimArmor(const Eigen::Matrix<double, 4, 4>& aims,
                                                   const Eigen::Matrix<double,4, 1>& Speed,
                                                   const Eigen::Matrix<double, 3, 1>& Gun);
    std::deque<Armor> FilterCenterArmor(const std::deque<std::array<ArmorPosi,2>>& armors_posis, const cv::Point3d& Gun, int num = 1);
    std::array<double,4> MeasureCov(const Eigen::Matrix<double, 4, 1>& view);
}


#endif