#ifndef POSEDETECTOR_HPP
#define POSEDETECTOR_HPP
#include "Armor.hpp"
#include <array>
#include <chrono>
#include <eigen3/Eigen/Core>

#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>
class PoseDetector
{
public:
    enum class State : int {ViewOne = 0, ViewUpAndDown = 1, ViewTwo = 2} state = State::ViewOne;

    struct TrackingArmor
    {
        enum class State : bool { Tracking = true, Lost=false } state = State::Lost;
        enum class Around : int { Left = 0, Right = 1, Unknow = 3} around = Around::Unknow;
        std::pair<Eigen::Vector3d, double> pose;//存储装甲板的中心坐标和yaw
        size_t ID = 0;
        double  yaw_min = -M_PI;
        std::chrono::steady_clock::time_point time;
    };


    explicit PoseDetector(const Eigen::Matrix<double, 3, 4>& Armors): Armors(Armors) {};
    std::pair< size_t, Eigen::Vector4d> operator ()(const ArmorPosi& armor,std::chrono::steady_clock::time_point now);
    std::vector< std::pair< size_t, Eigen::Vector4d> > operator ()(const std::vector<ArmorPosi>& armors, std::chrono::steady_clock::time_point now);
    
private:
    const Eigen::Matrix<double, 3, 4>& Armors;
    std::array< TrackingArmor, 2 > tracking_armors;

    double armor_lost_yawcos = std::cos(80.0 /180 * M_PI); 
};

#endif