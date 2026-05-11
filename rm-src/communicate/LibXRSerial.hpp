#pragma once
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/src/Geometry/Quaternion.h>
#include <thread>
#include "Data.hpp"
#include "FastQueue.hpp"
#include "LibXRRobotLink.hpp"
#include "mcudata.hpp"
#include <chrono>
#include <cstddef>

namespace io
{
template <size_t BufferSize = 50>
class LibXRSerial(LibXR::HardwareContainer& hw,
                  LibXR::ApplicationManager& appmgr)
{
private:
    struct Packet
    {
        std::chrono::steady_clock::time_point time;
        mcu::GimbalQuaternion quat;
    };
    FastQueueNoWait<Packet> Buffer_Que_(BufferSize);
    std::thread ReceiveThread_;

    static LibXR::Topic::Domain host_domain("host");

    static LibXR::Topic target_euler(
        "target_euler", sizeof(HostGimbalTarget), &host_domain, true);
    static LibXR::Topic fire_notify(
        "fire_notify", sizeof(HostFireNotify), &host_domain, true);

    static LibXR::Topic gimbal_gyro(
        "gimbal_gyro", sizeof(ImuSample), &host_domain, true);
    static LibXR::Topic gimbal_accl(
        "gimbal_accl", sizeof(ImuSample), &host_domain, true);
    static LibXR::Topic gimbal_quat(
        "gimbal_quat", sizeof(GimbalQuaternion), &host_domain, true);
    static LibXR::Topic robot_game_ref(
        "robot_game_ref", sizeof(RobotGameRefereeSummary), &host_domain, true);

    static SharedTopic shared_topic_rx(
        hw, appmgr, "DevC-USB", 4096,
        {{"gimbal_gyro", "host"},
         {"gimbal_accl", "host"},
         {"gimbal_quat", "host"},
         {"robot_game_ref", "host"}});

    static SharedTopicClient shared_topic_tx(
        hw, appmgr, "DevC-USB", 256,
        {{"target_euler", "host"},
         {"fire_notify", "host"}});

    LibXR::Topic::Domain& HostDomain()
    {
        static LibXR::Topic::Domain host_domain("host");
        return host_domain;
    }

public:

    LibXRSerial() 
    {
        float latest_temp = 0.0f;
        auto callback = LibXR::Topic::Callback::Create( 
            [&(this->Buffer_Que_)](bool,LibXR::RawData& data) -> void
                { 
                    Packet data;
                    data.time = std::chrono::steady_clock::now();
                    data.quat = *reinterpret_cast<mcu::GimbalQuaternion*>(data.addr_);
                    this->Buffer_Que_.push( data ); 
                }, 
                &latest_temp);
                
        LibXRSerial::gimbal_quat.RegisterCallback(callback);
    }


    bool ReadDate(Eigen::Quaterniond& quat,std::chrono::steady_clock::time_point& time)
    {
        Packet data;
        bool ret = this->Buffer_Que_.pop(data);

        if(ret){
            quat = Eigen::Quaterniond(data.quat.w,data.quat.x,data.quat.y,data.quat.z);
            time = data.time;
        }
        return ret;
    }
    void WriteDate(const HostGimbalTarget target, HostFireNotify fire_notify)
    {
        this->target_euler.Publish(target);
        this->fire_notify.Publish(fire_notify);
    }
};
}