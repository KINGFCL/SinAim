#include "LibXRRobotLink.hpp"

#include <algorithm>
#include <deque>
#include <mutex>

#include "SharedTopic.hpp"
#include "SharedTopicClient.hpp"
#include "app_framework.hpp"
#include "libxr.hpp"
#include "linux_uart.hpp"
#include "message.hpp"
#include "ramfs.hpp"
#include "thread.hpp"
#include "transform.hpp"

namespace io
{
namespace
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
}  // namespace

struct LibXRRobotLink::Impl
{
    struct QuatSample
    {
        cv::Quatd quat;
        timePoint time;
    };

    explicit Impl(std::size_t quat_buffer_size,
                  const char* vid,
                  const char* pid,
                  unsigned int baudrate)
        : quat_buffer_size(std::max<std::size_t>(1, quat_buffer_size)),
          peripherals{LibXR::Entry<LibXR::RamFS>({ramfs, {"ramfs"}})},
          host_domain("host")
    {
        gimbal_quat_topic = LibXR::Topic(
            LibXR::Topic::FindOrCreate<LibXR::Quaternion<float>>(
                "gimbal_quat", &host_domain));
        target_euler_topic = LibXR::Topic(
            LibXR::Topic::FindOrCreate<HostGimbalTarget>(
                "target_euler", &host_domain));
        fire_notify_topic = LibXR::Topic(
            LibXR::Topic::FindOrCreate<HostFireNotify>(
                "fire_notify", &host_domain));

        quat_callback = LibXR::Topic::Callback::Create(
            [](bool, Impl* self, LibXR::MicrosecondTimestamp,
               const LibXR::Quaternion<float>& quat)
            {
                self->onQuat(quat);
            },
            this);
        gimbal_quat_topic.RegisterCallback(quat_callback);

        devc_usb = std::make_unique<LibXR::LinuxUART>(
            vid, pid, baudrate, LibXR::UART::Parity::NO_PARITY, 8, 1, 80,
            8192);
        peripherals.Register(
            LibXR::Entry<LibXR::UART>({*devc_usb, {"DevC-USB"}}));

        shared_topic = std::make_unique<SharedTopic>(
            peripherals, appmgr, "DevC-USB", 4096,
            std::initializer_list<SharedTopic::TopicConfig>{
                {"gimbal_quat", "host"}});
        shared_topic_client = std::make_unique<SharedTopicClient>(
            peripherals, appmgr, "DevC-USB", 64,
            std::initializer_list<SharedTopicClient::TopicConfig>{
                {"target_euler", "host"}, {"fire_notify", "host"}});
    }

    void onQuat(const LibXR::Quaternion<float>& quat)
    {
        const QuatSample sample{
            cv::Quatd(static_cast<double>(quat.w()),
                      static_cast<double>(quat.x()),
                      static_cast<double>(quat.y()),
                      static_cast<double>(quat.z())),
            std::chrono::steady_clock::now()};

        std::lock_guard<std::mutex> lock(mutex);
        if (quat_samples.size() >= quat_buffer_size)
        {
            quat_samples.pop_front();
        }
        quat_samples.push_back(sample);
    }

    bool readQuat(cv::Quatd& quat, timePoint& data_time)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (quat_samples.empty())
        {
            return false;
        }

        const auto sample = quat_samples.front();
        quat_samples.pop_front();
        quat = sample.quat;
        data_time = sample.time;
        return true;
    }

    void publishCommand(float pitch, float yaw, bool fire)
    {
        HostGimbalTarget target{};
        target.pit = pitch;
        target.yaw = yaw;
        HostFireNotify fire_notify{fire};

        target_euler_topic.Publish(target);
        fire_notify_topic.Publish(fire_notify);
    }

    std::size_t quat_buffer_size;
    std::mutex mutex;
    std::deque<QuatSample> quat_samples;

    LibXR::RamFS ramfs;
    LibXR::HardwareContainer peripherals;
    LibXR::ApplicationManager appmgr;
    LibXR::Topic::Domain host_domain;
    LibXR::Topic gimbal_quat_topic;
    LibXR::Topic target_euler_topic;
    LibXR::Topic fire_notify_topic;
    LibXR::Topic::Callback quat_callback;
    std::unique_ptr<LibXR::LinuxUART> devc_usb;
    std::unique_ptr<SharedTopic> shared_topic;
    std::unique_ptr<SharedTopicClient> shared_topic_client;
};

LibXRRobotLink::LibXRRobotLink(std::size_t quat_buffer_size)
    : quat_buffer_size_(quat_buffer_size)
{
}

LibXRRobotLink::~LibXRRobotLink() = default;

bool LibXRRobotLink::start(const char* vid, const char* pid, unsigned int baudrate)
{
    if (impl_)
    {
        return true;
    }

    LibXR::PlatformInit();
    impl_ = std::make_unique<Impl>(quat_buffer_size_, vid, pid, baudrate);
    return true;
}

bool LibXRRobotLink::readQuat(cv::Quatd& quat, timePoint& data_time)
{
    return impl_ != nullptr && impl_->readQuat(quat, data_time);
}

void LibXRRobotLink::publishCommand(float pitch, float yaw, bool fire)
{
    if (impl_ == nullptr)
    {
        return;
    }
    impl_->publishCommand(pitch, yaw, fire);
}

}  // namespace io
