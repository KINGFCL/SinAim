#include "Target.hpp"
#include "Armor.hpp"
#include "EKFKalman.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <memory_resource>
#include <opencv2/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>


#define TargetDebug
//Debug

// std::array<RobotSize, 5> Robot::Size;

void Robot::Init(const std::vector<ArmorPosi>& armors)
{
    if(armors.empty()) return;
    
    // 装甲板类型检查
    if(static_cast<int>(armors[0].type) < 1) return;

    this->type = static_cast<Robot::Type>(static_cast<int>(armors[0].type));
    
    switch (armors.size()) {
        case 1:
        {
            Eigen::Vector3d center = 0.5*(armors[0].center.block<3,1>(0,0) + armors[0].center.block<3,1>(0,1));
            Eigen::Vector3d SCS = 0.5*(armors[0].SCS.block<3,1>(0,0) + armors[0].SCS.block<3,1>(0,1));
            this->InitLKF(center, SCS);
            break;
        }
        default:
            this->LKFToEKF(armors);
            break;
    }
}


void Robot::InitEKF(const Eigen::Vector3d& center, const Eigen::Vector3d& SCS, double yaw)
{
    // 初始化 EKF
    this->is_init = true;
    this->Mode = KalmanMode::EKF;
    this->ekfkalman.Init();

    // 初始化底盘中心点 (基于 0 号板和三角函数逆推)

    this->center(0) = center(0) - this->r * std::cos(yaw);
    this->center(1) = center(1) - this->r * std::sin(yaw);
    this->center(2) = center(2);

    // 初始化速度为 0
    this->Speed.setZero();
    
    // 半径
    this->Armors.block<1,4>(1,0) = Eigen::Matrix<double,1,4>{this->r,this->r,this->r,this->r};
    this->l_diff = 0.0;
    this->h_diff = 0.0;
    // 初始化 4 块装甲板的内部状态矩阵 [theta, radius, z]^T
    // 角度
    this->Armors(0,0) = yaw;
    this->Armors(0,1) = std::remainder(yaw + CV_PI/2.0,   CV_PI*2.0);
    this->Armors(0,2) = std::remainder(yaw + CV_PI,       CV_PI*2.0);
    this->Armors(0,3) = std::remainder(yaw + CV_PI*3.0/2.0, CV_PI*2.0);

    this->d_theta_1 = CV_PI/2.0;
    this->d_theta_2 = CV_PI;
    this->d_theta_3 = -CV_PI/2.0;

    // 高度 Z 坐标
    this->Armors(2,0) = this->Armors(2,2) = this->center(2);
    this->Armors(2,1) = this->Armors(2,3) = this->center(2);

    // 8. 重置所有视角状态
    this->View[0] = ArmorView::Visual;
    this->View[1] = ArmorView::Invisual;
    this->View[2] = ArmorView::Invisual;
    this->View[3] = ArmorView::Invisual;

}


void Robot::InitLKF(const Eigen::Vector3d& center, const Eigen::Vector3d& SCS)
{
    this->is_init = true;
    this->Mode = KalmanMode::LKF;
    this->lkfkalman.Init();
    
    this->center = center;
    this->Speed.setZero();
}

void Robot::LKFToEKF(const std::vector<ArmorPosi>& armors)
{
    if(armors.size()< 2 ) return;
    this->lkfkalman.Init();

    if(armors[0].yaw_abs[0]+armors[0].yaw_abs[1] <= armors[1].yaw_abs[0]+armors[1].yaw_abs[1])
    {
        if( std::remainder(armors[1].theta[0] - armors[0].theta[0], 2*CV_PI) < 0.0 )
            this->InitEKF(armors[0].center.block<3,1>(0,0), armors[0].SCS.block<3,1>(0,0), armors[0].yaw[0]);
        else
            this->InitEKF(armors[0].center.block<3,1>(0,1), armors[0].SCS.block<3,1>(0,1), armors[0].yaw[1]);
    } 
    else
    {
        if( std::remainder(armors[1].theta[0] - armors[0].theta[0], 2*CV_PI) < 0.0 )
            this->InitEKF(armors[1].center.block<3,1>(0,0), armors[1].SCS.block<3,1>(0,0), armors[1].yaw[0]);
        else
            this->InitEKF(armors[1].center.block<3,1>(0,1), armors[1].SCS.block<3,1>(0,1), armors[1].yaw[1]);
    }
}


void Robot::Update(const std::vector<ArmorPosi>& armors, double dt)
{
    if(armors.empty()) return;

    if(!is_init)
    {
        this->Init(armors);
        return;
    }

    switch (this->Mode) {
        case KalmanMode::LKF:
        {    //事件：直接观察到两个装甲板
            if(armors.size() > 1)
            {
                this->LKFToEKF(armors);
                return;
            }
            Eigen::Vector3d centerPredict = this->center + this->Speed.block<3,1>(0,0) * dt;

            Eigen::Vector3d ViewCenter = (armors[0].center.block<3,1>(0,0) + armors[0].center.block<3,1>(0,1) ) * 0.5;
            ViewCenter(2) = armors[0].center(2,0);

            double err = this->MatchErrorInLKF(centerPredict, ViewCenter);

            //事件：发生跳板
            if(err > this->matcherrthresh)
            {
                this->LKFToEKF(armors);
                return;
            }
            Eigen::Vector3d SCS = armors[0].SCS.block<3,1>(0,0);
            SCS(0) = ( armors[0].SCS(0,0) + armors[0].SCS(0,1) ) * 0.5;
            this->UpdateLKF( ViewCenter, SCS, dt);
            break;
        }
        case KalmanMode::EKF:
            this->UpdateEKF(armors, dt);
            break;
    }
}

void Robot::Update(double dt)
{
    switch (this->Mode) {
        case KalmanMode::LKF:
            this->UpdateLKF(dt);
            break;
        case KalmanMode::EKF:
            this->UpdateEKF(dt);
            break;
    }
}

void Robot::UpdateEKF(const std::vector<ArmorPosi>& armors, double dt)
{
    if(armors.empty()) return;

    if(armors.size() == 1)
    {
        this->OneArmor(armors[0], dt);
        return;
    }
    else 
    {
        this->TwoArmor(armors, dt);
        return;
    }

}

void Robot::UpdateEKF(double dt)
{

    //卡尔曼滤波
    //状态向量 State 为 14 维: [xc, yc, zc, vxc, vyc, vzc, theta_0, w, r,l,h,d_theta_1,d_theta_2,d_theta_3]
    Eigen::Matrix<double, 14, 1> State;
    State.block<3,1>(0,0) = this->center;
    State.block<3,1>(3,0) = this->Speed.block<3,1>(0,0);
    State(6,0) = this->Armors(0,0);
    State(7,0) = this->Speed(3,0);
    State(8,0) = this->Armors(1,0);
    State(9,0) = this->l_diff;
    State(10,0) = this->h_diff;
    State(11,0) = this->d_theta_1;
    State(12,0) = this->d_theta_2;
    State(13,0) = this->d_theta_3;

    Eigen::Matrix<double, 14, 1> ans = this->ekfkalman( State, dt );

    //更新中心点
    this->center = ans.block<3,1>(0,0);

    // 更新装甲板角度：以 ans(6,0) 为绝对基准推算所有板 (使用逆时针排布)
    this->Armors(0,0) = ans(6,0);
    this->Armors(0,1) = std::remainder(ans(6,0) + ans(11,0), CV_PI*2.0);
    this->Armors(0,2) = std::remainder(ans(6,0) + ans(12,0), CV_PI*2.0);
    this->Armors(0,3) = std::remainder(ans(6,0) + ans(13,0), CV_PI*2.0);
} 



void Robot::UpdateLKF(const Eigen::Vector3d& armorcenter, const Eigen::Vector3d& SCS, double dt)
{
    Eigen::Matrix<double, 6, 1> State;
    State.block<3,1>(0,0) = this->center;
    State.block<3,1>(3,0) = this->Speed.block<3,1>(0,0);

    Eigen::Matrix<double, 6, 1> ans = this->lkfkalman( State, armorcenter, SCS, dt );

    this->center = ans.block<3,1>(0,0);
    this->Speed.block<3,1>(0,0) = ans.block<3,1>(3,0);

    return;
}

void Robot::UpdateLKF(double dt)
{
    Eigen::Matrix<double, 6, 1> State;
    State.block<3,1>(0,0) = this->center;
    State.block<3,1>(3,0) = this->Speed.block<3,1>(0,0);

    Eigen::Matrix<double, 6, 1> ans = this->lkfkalman( State, dt );

    this->center = ans.block<3,1>(0,0);
    this->Speed.block<3,1>(0,0) = ans.block<3,1>(3,0);

    return;
}

double Robot::MatchErrorInLKF(const Eigen::Vector3d& armorcenter, const Eigen::Vector3d& ViewCenter)
{
    double distance = armorcenter.norm();

    double dot = armorcenter.dot(ViewCenter)/(distance*ViewCenter.norm());

    return std::acos(dot)*distance;
}

Robot::MatchAns Robot::MatchErrorInEKF(const ArmorPosi& armor, double dt)
{
    Eigen::Vector3d robot_center;
    Eigen::Matrix4d ans = this->Predict(dt, robot_center);
    Eigen::Vector3d armorcenter = 0.5 * ( armor.center.block<3,1>(0,0) + armor.center.block<3,1>(0,1) );
    double yaw_abs_view = 0.5*(armor.yaw_abs[0]+armor.yaw_abs[1]);
    std::array<size_t, 3> IDS;
    size_t ErrId;
    size_t CorrId;
    double distance = -1;

    for(size_t i = 0; i < 4; i++)
    {
        double norm = ans.block<3,1>(0,i).norm();

        if(norm > distance)
        {
            distance = norm;
            ErrId = i;
        }
    }
    IDS[0] = (ErrId+1)%4;
    IDS[1] = (ErrId+2)%4;
    IDS[2] = (ErrId+3)%4;

    double Err = 1e5;
    double norm = robot_center.norm();
    
    //匹配
    for(auto i : IDS)
    {
        double dot = ans.block<3,1>(0,i).dot(armorcenter)/(norm*armorcenter.norm());

        double yaw_abs = std::abs(std::remainder(ans(3,i) - std::atan2( ans(1,i), ans(0,i) ), CV_PI*2.0) ); 

        double theta_err = std::acos(dot);

        double yaw_err = std::abs( yaw_abs_view - yaw_abs);

        double err = (theta_err + yaw_err)*norm;
        if(err < Err)
        {
            Err = err;
            CorrId = i;
        }

    }
    
    size_t side;
    const Eigen::Vector3d& photocenter = armor.photocenter;
    const Eigen::Vector3d robot_center_cam = robot_center - photocenter;
    double robot_center_theta = std::atan2(robot_center_cam(1), robot_center_cam(0));

    Eigen::Vector3d armorcenter_cam = ans.block<3,1>(0,CorrId) - photocenter;

    double thetaArmor = std::atan2(armorcenter_cam(1), armorcenter_cam(0));
        
    if( std::remainder( robot_center_theta - thetaArmor, CV_PI*2.0) < 0.0 )
    {
        side = 0;
    }
    else
    {
        side = 1;
    }
    
    return MatchAns{CorrId, side, Err};
}
std::pair< Robot::MatchAns, Robot::MatchAns> Robot::MatchErrorInEKF(const std::vector<ArmorPosi>& armors, double dt)
{
    std::pair< Robot::MatchAns, Robot::MatchAns> ans;
    ans.first = this->MatchErrorInEKF(armors[0], dt);
    ans.second = this->MatchErrorInEKF(armors[1], dt);
    return ans;
}


void Robot::OneArmor(const ArmorPosi& armor, double dt)
{
    //装甲板匹配
    MatchAns matchAns = this->MatchErrorInEKF(armor, dt);
    size_t ID = matchAns.id;
    size_t side = matchAns.side;
    
    // 计算装甲板状态
    Eigen::Matrix<double, 4, 1> armorView {armor.center(0), armor.center(1), armor.center(2), armor.yaw[side]};

    //更新视角
    this->View[(ID+1)%4] = this->View[(ID+2)%4] = this->View[(ID+3)%4] = ArmorView::Invisual;
    this->View[ID] = ArmorView::Visual;


    //卡尔曼滤波
    //状态向量 State 为 14 维: [xc, yc, zc, vxc, vyc, vzc, theta_0, w, r,l,h,d_theta_1,d_theta_2,d_theta_3]
    Eigen::Matrix<double, 14, 1> State;
    State.block<3,1>(0,0) = this->center;
    State.block<3,1>(3,0) = this->Speed.block<3,1>(0,0);
    State(6,0) = this->Armors(0,0);
    State(7,0) = this->Speed(3,0);
    State(8,0) = this->Armors(1,0);
    State(9,0) = this->l_diff;
    State(10,0) = this->h_diff;
    State(11,0) = this->d_theta_1;
    State(12,0) = this->d_theta_2;
    State(13,0) = this->d_theta_3;

    Eigen::Matrix<double, 14, 1> ans = this->ekfkalman(State, armorView, armor.SCS.block<3,1>(0,side), armor.yaw_abs[side], ID,dt);
    
    //更新l,h
    this->l_diff = ans(9,0);
    this->h_diff = ans(10,0);

    //更新角度
    this->d_theta_1 = ans(11,0);
    this->d_theta_2 = ans(12,0);
    this->d_theta_3 = ans(13,0);

    //更新Robot信息
    this->Speed.block<3,1>(0,0) = ans.block<3,1>(3,0);
    this->Speed(3,0) = ans(7,0);

    //更新中心点
    this->center = ans.block<3,1>(0,0);

    //更新半径
    this->Armors(1,0) = this->Armors(1,2) = ans(8,0);
    this->Armors(1,1) = this->Armors(1,3) = ans(8,0) + this->l_diff;

    //更新高度
    this->Armors(2,0) = this->Armors(2,2) = ans(2,0);
    this->Armors(2,1) = this->Armors(2,3) = ans(2,0) + this->h_diff;

    // 更新装甲板角度：以 ans(6,0) 为绝对基准推算所有板 (使用逆时针排布)
    this->Armors(0,0) = ans(6,0);
    this->Armors(0,1) = std::remainder(ans(6,0) + ans(11,0), CV_PI*2.0);
    this->Armors(0,2) = std::remainder(ans(6,0) + ans(12,0), CV_PI*2.0);
    this->Armors(0,3) = std::remainder(ans(6,0) + ans(13,0), CV_PI*2.0);
}

void Robot::TwoArmor(const std::vector<ArmorPosi>& armors, double dt)
{
    std::array< Eigen::Matrix<double, 4, 1>, 2 > ArmorStates;
    std::array<size_t, 2> IDS;
    std::array<size_t, 2> side;

    // ==========================================
    // 1. 提取观测并匹配 ID
    // ==========================================
    std::pair<MatchAns,MatchAns> matchAns = this->MatchErrorInEKF(armors, dt);
    IDS[0] = matchAns.first.id;
    IDS[1] = matchAns.second.id;

    side[0] = matchAns.first.side;
    side[1] = matchAns.second.side;

    ArmorStates[0].block<3,1>(0,0) = armors[0].center.block<3,1>(0,side[0]);
    ArmorStates[1].block<3,1>(0,0) = armors[1].center.block<3,1>(0,side[1]);

    ArmorStates[0](3,0) = armors[0].yaw[side[0]];
    ArmorStates[1](3,0) = armors[1].yaw[side[1]];

    if(IDS[0] == IDS[1]) {
        this->Update(dt);
        return;
    }

    // ==========================================
    // 2. 更新视角 View
    // ==========================================
    for(int i = 0; i < 4; i++) {
        this->View[i] = ArmorView::Invisual;
    }

    this->View[IDS[0]] = ArmorView::Visual;
    this->View[IDS[1]] = ArmorView::Visual;


    // ==========================================
    // 3. 组装 14 维先验状态 (绝对锚定 0 号装甲板!)
    // ==========================================
    Eigen::Matrix<double, 14, 1> State;
    State.block<3,1>(0,0) = this->center;
    State.block<3,1>(3,0) = this->Speed.block<3,1>(0,0);
    State(6,0) = this->Armors(0,0);  // 绝对基准：永远传 0 号板角度
    State(7,0) = this->Speed(3,0);
    State(8,0) = this->Armors(1,0);  // 绝对基准：永远传 0 号板半径
    State(9,0) = this->l_diff;
    State(10,0) = this->h_diff;
    State(11,0) = this->d_theta_1;
    State(12,0) = this->d_theta_2;
    State(13,0) = this->d_theta_3;

    // ==========================================
    // 卡尔曼滤波 (一次喂入多个观测值)
    // ==========================================
    std::array<Eigen::Vector3d, 2> SCSs;
    std::array<double, 2>yaws;

    if( ((IDS[0] + 1) % 4) != IDS[1] )
    {
        Eigen::Matrix<double, 4, 1> P_ = ArmorStates[0];
        ArmorStates[0] = ArmorStates[1];
        ArmorStates[1] = P_;

        size_t id_ = IDS[0];
        IDS[0] = IDS[1];
        IDS[1] = id_;
        
        SCSs[0] = armors[1].SCS.block<3,1>(0,side[1]);
        SCSs[1] = armors[0].SCS.block<3,1>(0,side[0]);

        yaws[0] = armors[1].yaw_abs[side[1]];
        yaws[1] = armors[0].yaw_abs[side[0]];
    }else {
        SCSs[0] = armors[0].SCS.block<3,1>(0,side[0]);
        SCSs[1] = armors[1].SCS.block<3,1>(0,side[1]);

        yaws[0] = armors[0].yaw_abs[side[0]];
        yaws[1] = armors[1].yaw_abs[side[1]];
    }
    Eigen::Matrix<double, 10, 1> StateViews;
    StateViews.block<4,1>(0,0) = ArmorStates[0];
    StateViews.block<4,1>(4,0) = ArmorStates[1];
    StateViews(8,0) = ArmorStates[1](2,0) - ArmorStates[0](2,0);
    StateViews(9,0) = std::remainder(ArmorStates[1](3,0) - ArmorStates[0](3,0), 2*CV_PI);

    Eigen::Matrix<double, 14, 1> ans = this->ekfkalman(State, StateViews, SCSs[0], SCSs[1], yaws[0], yaws[1], IDS[0], dt);
    
    // 更新 l, h
    this->l_diff = ans(9,0);
    this->h_diff = ans(10,0);

    // 更新角度差
    this->d_theta_1 = ans(11,0);
    this->d_theta_2 = ans(12,0);
    this->d_theta_3 = ans(13,0);

    // 更新中心点与速度
    this->center = ans.block<3,1>(0,0);
    this->Speed.block<3,1>(0,0) = ans.block<3,1>(3,0);
    this->Speed(3,0) = ans(7,0);

    // 更新半径：0,2 是基础半径 r；1,3 是侧面半径 r+l
    this->Armors(1,0) = this->Armors(1,2) = ans(8,0);
    this->Armors(1,1) = this->Armors(1,3) = ans(8,0) + this->l_diff;

    // 更新高度：0,2 是基础高度 zc；1,3 是侧面高度 zc+h
    this->Armors(2,0) = this->Armors(2,2) = ans(2,0);
    this->Armors(2,1) = this->Armors(2,3) = ans(2,0) + this->h_diff;

    // 更新装甲板角度：以 ans(6,0) 为绝对基准推算所有板 (使用逆时针排布)
    this->Armors(0,0) = ans(6,0);
    this->Armors(0,1) = std::remainder(ans(6,0) + ans(11,0), CV_PI*2.0);
    this->Armors(0,2) = std::remainder(ans(6,0) + ans(12,0), CV_PI*2.0);
    this->Armors(0,3) = std::remainder(ans(6,0) + ans(13,0), CV_PI*2.0);
}




Eigen::Matrix<double, 4, 4> Robot::Predict(double dt)
{
    double& w = this->Speed(3,0);

    Eigen::Matrix<double, 4, 4> ans;

    ans(3,0) = std::remainder(this->Armors(0,0) + w*dt, 2.0 * CV_PI);
    ans(3,1) = std::remainder(this->Armors(0,1) + w*dt, 2.0 * CV_PI);
    ans(3,2) = std::remainder(this->Armors(0,2) + w*dt, 2.0 * CV_PI);
    ans(3,3) = std::remainder(this->Armors(0,3) + w*dt, 2.0 * CV_PI);

    //旋转后的位置
    ans.block<2,1>(0,0) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,0)*std::cos(ans(3,0)), this->center(1) + this->Armors(1,0)*std::sin(ans(3,0))};
    ans.block<2,1>(0,1) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,1)*std::cos(ans(3,1)), this->center(1) + this->Armors(1,1)*std::sin(ans(3,1))};
    ans.block<2,1>(0,2) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,2)*std::cos(ans(3,2)), this->center(1) + this->Armors(1,2)*std::sin(ans(3,2))};
    ans.block<2,1>(0,3) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,3)*std::cos(ans(3,3)), this->center(1) + this->Armors(1,3)*std::sin(ans(3,3))};
    
    //加上平移
    ans.block<1,4>(2,0) = this->Armors.block<1,4>(2,0);
    auto move = this->Speed.block<3,1>(0,0)*dt;
    ans.block<3,1>(0,0) += move;
    ans.block<3,1>(0,1) += move;
    ans.block<3,1>(0,2) += move;
    ans.block<3,1>(0,3) += move;

    return ans;
}

Eigen::Matrix<double, 4, 4> Robot::Predict(double dt, Eigen::Vector3d& Center)
{
    double& w = this->Speed(3,0);

    Eigen::Matrix<double, 4, 4> ans;

    ans(3,0) = std::remainder(this->Armors(0,0) + w*dt, 2.0 * CV_PI);
    ans(3,1) = std::remainder(this->Armors(0,1) + w*dt, 2.0 * CV_PI);
    ans(3,2) = std::remainder(this->Armors(0,2) + w*dt, 2.0 * CV_PI);
    ans(3,3) = std::remainder(this->Armors(0,3) + w*dt, 2.0 * CV_PI);

    //旋转后的位置
    ans.block<2,1>(0,0) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,0)*std::cos(ans(3,0)), this->center(1) + this->Armors(1,0)*std::sin(ans(3,0))};
    ans.block<2,1>(0,1) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,1)*std::cos(ans(3,1)), this->center(1) + this->Armors(1,1)*std::sin(ans(3,1))};
    ans.block<2,1>(0,2) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,2)*std::cos(ans(3,2)), this->center(1) + this->Armors(1,2)*std::sin(ans(3,2))};
    ans.block<2,1>(0,3) = Eigen::Matrix<double,2,1>{this->center(0) + this->Armors(1,3)*std::cos(ans(3,3)), this->center(1) + this->Armors(1,3)*std::sin(ans(3,3))};
    
    //加上平移
    ans.block<1,4>(2,0) = this->Armors.block<1,4>(2,0);
    Eigen::Matrix<double, 3, 1> move = this->Speed.block<3,1>(0,0)*dt;
    ans.block<3,1>(0,0) += move;
    ans.block<3,1>(0,1) += move;
    ans.block<3,1>(0,2) += move;
    ans.block<3,1>(0,3) += move;

    Center = this->center + move;

    return ans;
}



void Robot::Clear()
{
    this->is_init = false;
}

Eigen::Matrix<double, 3, 1> Robot::Rotate(
    const Eigen::Matrix<double, 3, 1>& Point, 
    double angle )
    {
    // 2. 去中心化（平移）
    Eigen::Matrix<double, 3, 1> P = Point - this->center;

    // 3. 构造旋转
    // AngleAxisd 接受任何 derived from MatrixBase 的 3x1 向量
    Eigen::AngleAxisd rotation_vector{angle, this->axis};

    // 4. 旋转并还原中心
    return (rotation_vector * P) + this->center;
}




// /**
//  * @brief 如果输入的装甲板数量为2，函数将更新Robot的尺寸
//  * @param armors 输入的装甲板位置，必须包含两个装甲板
//  * @note 如果输入的装甲板数量不为2，函数不会执行任何操作
//  */
//  void Robot::SolveRobotSize(std::vector<ArmorPosi>& armors)
// {
//     //如果输入的装甲板数量不为2，则函数不会执行任何操作
//     if(armors.size() != 2 || armors[0].type != armors[1].type) return;
    
//     #ifdef TargetDebug
//     //观测到的装甲板朝向夹角
//     double theta_debug = std::acos((armors[0].toward.dot(armors[1].toward))/(cv::norm(armors[0].toward)*cv::norm(armors[1].toward)));
//     theta_debug = (theta_debug/CV_PI)*180.0;
//     std::cout <<"armor theta: "<< theta_debug << "\n";
//     #endif

//     //计算中心轴向量
//     cv::Point3d axis_ = armors[0].toward.cross(armors[1].toward);
//     axis_ = axis_ / cv::norm(axis_);//单位化

//     //统一方向
//     double cos0 = axis_.dot(cv::Point3d(0,-1,0));
//     if(cos0 < 0) axis_ = -axis_;


// //计算半径和高度差
//     cv::Point3d armor1_face = axis_.cross(armors[0].toward);
//     cv::Point3d armor2_face = axis_.cross(armors[1].toward);
//     armor1_face = armor1_face / cv::norm(armor1_face);//单位化
//     armor2_face = armor2_face / cv::norm(armor2_face);//单位化

//     cv::Point3d armorOneToTwo = armors[1].posi - armors[0].posi;

//     double b1 = armorOneToTwo.dot(armor1_face);
//     double b2 = armorOneToTwo.dot(armor2_face);


//     double k = armor1_face.dot(armor2_face);
    
//     double deno = 1 - k*k;

// //计算r
//     double ArmorOneR = std::abs( ( b1 - b2 * k ) / deno );
//     double ArmorTwoR = std::abs( ( b1 * k - b2 ) / deno );

// //判断半径大小,并更新Robot尺寸
//     if(ArmorOneR < ArmorTwoR) 
//     {
//         armors[0].radius = ArmorPosi::Radius::Short;
//         armors[1].radius = ArmorPosi::Radius::Long;
//         Robot::Size[static_cast<int>(armors[0].type)-1](ArmorOneR, ArmorTwoR);
//     }
//     else
//     {
//         armors[0].radius = ArmorPosi::Radius::Long;
//         armors[1].radius = ArmorPosi::Radius::Short;
//         Robot::Size[static_cast<int>(armors[0].type)-1](ArmorTwoR, ArmorOneR);
//     }
// }



