#ifndef POSEDETECTOR_HPP
#define POSEDETECTOR_HPP
#include "Armor.hpp"
#include "CircularQueue.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <eigen3/Eigen/Core>

#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>
class PoseDetector
{
public:
    enum class State : int {ViewOne = 0, ViewUpAndDown = 1, ViewTwo = 2} state = State::ViewOne;
    enum class FindErr : bool {Yes = true, No = false} finderr = FindErr::No;

    struct TrackingArmor
    {
        CircularQueue< std::pair<double, double> > yawAndTime{30};
        std::pair<Eigen::Vector3d, double> pose;//存储装甲板的中心坐标和yaw
        double yaw_abs = 0;
        size_t ID;
        std::chrono::steady_clock::time_point StartTime;

        enum class State : bool { Tracking = true, Lost=false } state;
        enum class Around : int { Left = 0, Right = 1, Unknow = 3} Startaround;
        bool isFlipped = false;

        TrackingArmor():ID(0),StartTime(std::chrono::steady_clock::now()),state(State::Lost),Startaround(Around::Unknow){};
        void Init(size_t ID, ArmorPosi armor, std::chrono::steady_clock::time_point time, State state, Around start_around)
        {
            this->Clear();
            this->ID = ID;
            this->StartTime = time;
            this->state = state;
            this->Startaround = start_around;

            switch (start_around) {
                case Around::Unknow:
                    this->pose.first = armor.center.block<3,1>(0,0);
                    this->pose.second = armor.yaw[0];
                    this->yaw_abs = std::abs(std::remainder(armor.yaw[0]-std::atan2(armor.center(1,0),armor.center(0,0)),2*M_PI));
                    this->yawAndTime.enQueue(std::make_pair(yaw_abs,0));
                    this->isFlipped = false;
                    break;
                case Around::Left:
                    this->pose.first = armor.center.block<3,1>(0,0);
                    this->pose.second = armor.yaw[0];
                    this->yaw_abs = std::abs(std::remainder(armor.yaw[0]-std::atan2(armor.center(1,0),armor.center(0,0)),2*M_PI));
                    this->yawAndTime.enQueue(std::make_pair(yaw_abs,0));
                    this->isFlipped = false;
                    break;
                case Around::Right:
                    this->pose.first = armor.center.block<3,1>(0,1);
                    this->pose.second = armor.yaw[1];
                    double yaw_abs = std::abs(std::remainder(armor.yaw[1]-std::atan2(armor.center(1,0),armor.center(0,0)),2*M_PI));
                    this->yawAndTime.enQueue(std::make_pair(yaw_abs,0));
                    this->isFlipped = false;                    
                    break;
            }
        }
        void Clear(){ this->state = State::Lost; this->yawAndTime.clear(); }

        void operator()(const ArmorPosi& armor, std::chrono::steady_clock::time_point now);

        bool leastSquaresFit(double& slope, double& intercept);
    };


    explicit PoseDetector(){};
    std::pair< size_t, Eigen::Vector4d> operator ()(const ArmorPosi& armor,std::chrono::steady_clock::time_point now);
    std::vector< std::pair< size_t, Eigen::Vector4d> > operator ()(const std::vector<ArmorPosi>& armors, std::chrono::steady_clock::time_point now);
    
private:
    std::array< TrackingArmor, 2 > tracking_armors;
    double matchRadian = 10;//cm
};
#endif