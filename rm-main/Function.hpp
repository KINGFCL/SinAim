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
#include <vector>

//IMU与图像配对线程逻辑
namespace rm
{
    void IMUAndImageMatchFunction(io::HikCamera& Hik, io::RTSerial<Packet>& ser,FastQueue<FrameData>& Frames);
    void SendMessageToRobot(io::RTSerial<Packet>& ser, float pitch, float yaw, bool fire);
    Eigen::Matrix<double, 3, 1> ChooseBestAimArmor(const Eigen::Matrix<double, 4, 4>& aims,
                                                   const Eigen::Matrix<double,4, 1>& Speed,
                                                   const Eigen::Matrix<double, 3, 1>& Gun);
    std::deque<Armor> FilterCenterArmor(const std::deque<std::array<ArmorPosi,2>>& armors_posis, const cv::Point3d& Gun, int num = 1);
    double SolveDt(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end, double pic);
    
    std::vector<YoloArmor> MatchYoloAndOpenCV(const std::deque<Armor>& armors,const std::vector<YoloArmor>& yolo_armors);
    bool checkPitchYawDeviation(const cv::Quatd& quad, 
                                double pitch_C, double yaw_C, 
                                double pitch_thresh, double yaw_thresh);
}


#endif