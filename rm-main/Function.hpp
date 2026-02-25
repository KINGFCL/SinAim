#ifndef INCLUDE_FUNCTION_HPP
#define INCLUDE_FUNCTION_HPP

#include "include/Data.hpp"
#include "include/HikCamera.hpp"
#include "include/RTSerial.hpp"
#include "include/fastqueue.hpp"

//IMU与图像配对线程逻辑
namespace rm
{
    void IMUAndImageMatchFunction(io::HikCamera& Hik, io::RTSerial<Packet>& ser,FastQueue<FrameData>& Frames);
    void SendMessageToRobot(io::RTSerial<Packet>& ser, float pitch, float yaw, bool fire);
   
}



#endif