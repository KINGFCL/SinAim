#include "PoseDetector.hpp"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <utility>
#include <vector>

void PoseDetector::TrackingArmor::operator()(const ArmorPosi& armor, std::chrono::steady_clock::time_point now)
{
    if(this->Startaround == PoseDetector::TrackingArmor::Around::Unknow)
    {
        this->pose.first = armor.center.block<3,1>(0,0);
        this->pose.second = armor.yaw[0];
        return;
    }
    if(isFlipped)
    {
        switch (this->Startaround) {
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

    int yaw_index = static_cast<int>(this->Startaround);
    double yaw_may = armor.yaw[yaw_index];

    double center_theta = std::atan2(armor.center(1,yaw_index),armor.center(0,yaw_index));

    double yaw_abs = std::abs(std::remainder(yaw_may-center_theta,2*M_PI));

    //入队列
    this->yawAndTime.enQueue(std::make_pair(yaw_abs, t_seconds));

    double slope, intercept;
    bool solveOK = this->leastSquaresFit(slope, intercept);
    

    //如果求解不了说明数据太少。如果斜率接近0说明静止，如果斜率>0说明没有往中心旋转
    if( (!solveOK) || slope < -1e-5)
    {
        this->pose.first = armor.center.block<3,1>(0,yaw_index);
        this->pose.second = armor.yaw[yaw_index];
        return;
    }

    double flip_time = (-intercept) / slope;


    //当前时间已经发生翻转
    if(flip_time <= t_seconds)
    {
        this->isFlipped = true;
        this->pose.first = armor.center.block<3,1>(0,1-yaw_index);
        this->pose.second = armor.yaw[1-yaw_index];
        return;
    }

    //没有翻转过去
    this->pose.first = armor.center.block<3,1>(0,yaw_index);
    this->pose.second = armor.yaw[yaw_index];
}

bool PoseDetector::TrackingArmor::leastSquaresFit(double& slope, double& intercept)
{
    if(this->yawAndTime.isEmpty()) return false;

    size_t index = 0;
    double yaw_abs_min = this->yawAndTime[0].first;

    for(size_t i = 1; i < yawAndTime.size(); i++)
    {
        if(yawAndTime[i].first < yaw_abs_min)
        {
            yaw_abs_min = yawAndTime[i].first;
            index = i;
        }
    }

    // 至少需要两个点才能拟合直线
    size_t n = index+1;
    if (n < 2) {
        return false; 
    }

    double sumX = 0.0;
    double sumY = 0.0;
    double sumXY = 0.0;
    double sumX2 = 0.0;

    // 遍历所有数据点，计算需要的各项和
    for (size_t i = 0; i < n; i++) {

        const std::pair<double, double>& point = this->yawAndTime[i];

        double x = point.first;
        double y = point.second;
        
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
    }

    // 根据公式计算分母: n * Σ(x^2) - (Σx)^2
    double denominator = ((double)n) * sumX2 - sumX * sumX;

    // 检查分母是否趋近于 0（即所有点的 x 坐标几乎相同，是一条垂直于 x 轴的线）
    // 浮点数比较不建议直接用 == 0，使用极小值 1e-9 作为容差
    if (std::abs(denominator) < 1e-9) {
        return false; 
    }

    // 计算斜率 a = (n * Σ(xy) - Σx * Σy) / 分母
    slope = (n * sumXY - sumX * sumY) / denominator;
    
    // 计算截距 b = (Σy - a * Σx) / n
    intercept = (sumY - slope * sumX) / n;

    return true;
}

std::vector< std::pair< size_t, Eigen::Vector4d> > PoseDetector::operator ()(const std::vector<ArmorPosi>& armors, std::chrono::steady_clock::time_point now)
{
    if(armors.empty()) return std::vector< std::pair< size_t, Eigen::Vector4d> >();
    if(armors.size() == 1){
        auto ans = this->operator()(armors[0], now);
        return std::vector< std::pair< size_t, Eigen::Vector4d> >{ans};
    }
    //两个板的情况
    this->state = State::ViewTwo;
    //先和tracking_armors匹配



}

std::pair< size_t, Eigen::Vector4d> PoseDetector::operator ()(const ArmorPosi& armor, std::chrono::steady_clock::time_point now)
{   
    Eigen::Vector3d center = armor.center.block<3,1>(0,0)+armor.center.block<3,1>(0,1);

    double distence = center.norm();

    double diff = 1e5;

    size_t index = 0;
    
    for(size_t i = 0; i < this->tracking_armors.size(); i++)
    {
        if(tracking_armors[i].state == TrackingArmor::State::Lost) { continue; }

        double tracking_distence = tracking_armors[i].pose.first.norm();

        double dot = center.dot(tracking_armors[i].pose.first);

        double theta_diff =  std::acos(dot/(distence*tracking_distence));

        double r = (distence+tracking_distence)/2;

        double now_diff = r*theta_diff;

        if(now_diff < diff)
        {
            diff = now_diff;
            index = i;
        }
    }

    if(diff <= this->matchRadian)
    {
        this->tracking_armors[index](armor, now);
        Eigen::Vector4d ans;
        ans.block<3,1>(0,0) = this->tracking_armors[index].pose.first;
        ans(3,0) = this->tracking_armors[index].pose.second;
        return std::make_pair(this->tracking_armors[index].ID, ans);
    }

    //出现新板的情况
    this->state = State::ViewUpAndDown;

    double yaw_absmax = -1;
    for(size_t i = 0; i < this->tracking_armors.size(); i++)
    {
        if(tracking_armors[i].state == TrackingArmor::State::Lost) { index = i; break; }

        if(yaw_absmax < tracking_armors[i].yaw_abs)
        {
            yaw_absmax = tracking_armors[i].yaw_abs;
            index = i;
        }
    }

    double face_theta_other = 0;
    double face_theta_now = 0; 
    {
        Eigen::Vector3d face_other = this->tracking_armors[1-index].pose.first;
        face_theta_other = std::atan2(face_other(1,0), face_other(0,0));

        face_theta_now = std::atan2(center(1,0), center(0,0));
    }

    double face_diff = std::remainder(face_theta_other - face_theta_now, 2*M_PI);

    if(face_diff > 0)
    {
        this->tracking_armors[index].Init((this->tracking_armors[1-index].ID+3)%4, armor, now, TrackingArmor::State::Tracking, TrackingArmor::Around::Left);
        Eigen::Vector4d ans;
        ans.block<3,1>(0,0) = this->tracking_armors[index].pose.first;
        ans(3,0) = this->tracking_armors[index].pose.second;

        if(this->tracking_armors[1-index].Startaround == TrackingArmor::Around::Left) {this->finderr = FindErr::Yes; this->tracking_armors[1-index].isFlipped = true;}
        return std::make_pair(this->tracking_armors[index].ID, ans);
    }

    this->tracking_armors[index].Init((this->tracking_armors[1-index].ID+3)%4, armor, now, TrackingArmor::State::Tracking, TrackingArmor::Around::Right);
    Eigen::Vector4d ans;
    ans.block<3,1>(0,0) = this->tracking_armors[index].pose.first;
    ans(3,0) = this->tracking_armors[index].pose.second;
    return std::make_pair(this->tracking_armors[index].ID, ans);
}