#pragma once

#include <cstdint>

#include "SharedTopic/SharedTopic.hpp"
#include "SharedTopicClient/SharedTopicClient.hpp"
#include "app_framework.hpp"

namespace io::mcu
{

struct HostGimbalTarget
{
    float rol{};
    float pit{};
    float yaw{};
    float rol_dot{};
    float pit_dot{};
    float yaw_dot{};
    float rol_ddot{};
    float pit_ddot{};
    float yaw_ddot{};
};

struct HostFireNotify
{
    bool isfire{};
};

struct ImuSample
{
    float x{};
    float y{};
    float z{};
};

struct GimbalQuaternion
{
    float x{};
    float y{};
    float z{};
    float w{1.0F};
};

struct [[gnu::packed]] RobotGameRefereeStatus
{
    uint8_t robot_id{};
    uint8_t robot_level{};
    uint16_t remain_hp{};
    uint16_t max_hp{};
    uint16_t shooter_cooling_value{};
    uint16_t shooter_heat_limit{};
    uint16_t chassis_power_limit{};
    uint8_t power_gimbal_output : 1;
    uint8_t power_chassis_output : 1;
    uint8_t power_launcher_output : 1;
};

struct [[gnu::packed]] RobotGameRefereeGame
{
    uint8_t game_type : 4;
    uint8_t game_progress : 4;
    uint16_t stage_remain_time{};
    uint64_t sync_time_stamp{};
};

struct [[gnu::packed]] RobotGameRefereeLauncher
{
    uint8_t bullet_type{};
    uint8_t launcher_id{};
    uint8_t bullet_freq{};
    float bullet_speed{};
};

struct [[gnu::packed]] RobotGameRefereeSummary
{
    RobotGameRefereeStatus robot_status{};
    RobotGameRefereeGame game_status{};
    RobotGameRefereeLauncher launcher_data{};
};

static_assert(sizeof(RobotGameRefereeStatus) == 13);
static_assert(sizeof(RobotGameRefereeGame) == 11);
static_assert(sizeof(RobotGameRefereeLauncher) == 7);
static_assert(sizeof(RobotGameRefereeSummary) == 31);
static_assert(sizeof(ImuSample) == 12);
static_assert(sizeof(GimbalQuaternion) == 16);

inline void InitMcuCommunication(LibXR::HardwareContainer& hw,
                                 LibXR::ApplicationManager& appmgr)
{
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
}

}  // namespace io::mcu
