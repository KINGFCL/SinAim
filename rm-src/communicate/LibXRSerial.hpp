#pragma once

#include <chrono>
#include <cstddef>
#include "SharedTopic/SharedTopic.hpp"
#include "SharedTopicClient/SharedTopicClient.hpp"
#include <Eigen/Geometry>

#include "FastQueue.hpp"
#include "mcudata.hpp"

namespace io
{

template <std::size_t BufferSize = 50>
class LibXRSerial
{
 public:
  LibXRSerial(LibXR::HardwareContainer& hw, LibXR::ApplicationManager& appmgr)
      : buffer_queue_(BufferSize),
        host_domain_("host"),
        target_euler_("target_euler", sizeof(mcu::HostGimbalTarget), &host_domain_,
                      true),
        fire_notify_("fire_notify", sizeof(mcu::HostFireNotify), &host_domain_, true),
        gimbal_gyro_("gimbal_gyro", sizeof(mcu::ImuSample), &host_domain_, true),
        gimbal_accl_("gimbal_accl", sizeof(mcu::ImuSample), &host_domain_, true),
        gimbal_quat_("gimbal_quat", sizeof(mcu::GimbalQuaternion), &host_domain_,
                     true),
        robot_game_ref_("robot_game_ref", sizeof(mcu::RobotGameRefereeSummary),
                        &host_domain_, true),
        shared_topic_rx_(hw, appmgr, "DevC-USB", 4096,
                         {{"gimbal_gyro", "host"},
                          {"gimbal_accl", "host"},
                          {"gimbal_quat", "host"},
                          {"robot_game_ref", "host"}}),
        shared_topic_tx_(hw, appmgr, "DevC-USB", 256,
                         {{"target_euler", "host"}, {"fire_notify", "host"}}),
        gimbal_quat_callback_(LibXR::Topic::Callback::Create(
            [](bool, LibXRSerial* self, const mcu::GimbalQuaternion& quat) {
              self->OnGimbalQuaternion(quat);
            },
            this))
  {
    gimbal_quat_.RegisterCallback(gimbal_quat_callback_);
  }

  bool ReadData(Eigen::Quaterniond& quat,
                std::chrono::steady_clock::time_point& time)
  {
    Packet packet;
    const bool ret = buffer_queue_.pop(packet);
    if (ret)
    {
      quat = Eigen::Quaterniond(packet.quat.w, packet.quat.x, packet.quat.y,
                                packet.quat.z);
      time = packet.time;
    }
    return ret;
  }

  bool ReadDate(Eigen::Quaterniond& quat,
                std::chrono::steady_clock::time_point& time)
  {
    return ReadData(quat, time);
  }

  void WriteData(mcu::HostGimbalTarget target, mcu::HostFireNotify fire_notify)
  {
    target_euler_.Publish(target);
    fire_notify_.Publish(fire_notify);
  }

  void WriteDate(mcu::HostGimbalTarget target, mcu::HostFireNotify fire_notify)
  {
    WriteData(target, fire_notify);
  }

 private:
  struct Packet
  {
    std::chrono::steady_clock::time_point time;
    mcu::GimbalQuaternion quat;
  };

  void OnGimbalQuaternion(const mcu::GimbalQuaternion& quat)
  {
    Packet packet;
    packet.time = std::chrono::steady_clock::now();
    packet.quat = quat;
    buffer_queue_.push(packet);
  }

  FastQueueNoWait<Packet> buffer_queue_;

  LibXR::Topic::Domain host_domain_;

  LibXR::Topic target_euler_;
  LibXR::Topic fire_notify_;
  LibXR::Topic gimbal_gyro_;
  LibXR::Topic gimbal_accl_;
  LibXR::Topic gimbal_quat_;
  LibXR::Topic robot_game_ref_;

  SharedTopic shared_topic_rx_;
  SharedTopicClient shared_topic_tx_;
  LibXR::Topic::Callback gimbal_quat_callback_;
};

}  // namespace io
