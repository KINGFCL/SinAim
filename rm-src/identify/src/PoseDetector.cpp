#include "PoseDetector.hpp"
#include <chrono>
#include <utility>

void PoseDetector::TrackingArmor::operator()(const ArmorPosi& armor, std::chrono::steady_clock::time_point now)
{
    if(this->around == PoseDetector::TrackingArmor::Around::Unknow)
    {
        this->pose.first = armor.center.block<3,1>(0,0);
        this->pose.second = armor.yaw[0];
        return;
    }
    if(isFlipped)
    {
        switch (this->around) {
            case PoseDetector::TrackingArmor::Around::Left:
                this->pose.first = armor.center.block<3,1>(0,1);
                this->pose.second = armor.yaw[1];
                break;
            case PoseDetector::TrackingArmor::Around::Right:
                this->pose.first = armor.center.block<3,1>(0,0);
                this->pose.second = armor.yaw[0];
                break;
            default:
                break;
        }
        return;
    }

    //没有翻转过去，去判断翻转了没有
    double t_seconds = std::chrono::duration<double>(now - this->StartTime).count();

    int yaw_index = static_cast<int>(this->around);
    double yaw_may = armor.yaw[yaw_index];

    double center_theta = std::atan2(armor.center(1,yaw_index),armor.center(0,yaw_index));


    std::pair<double, double> yawAndtime_pair = std::make_pair(armor.yaw[yaw_index], t_seconds);
    



}
