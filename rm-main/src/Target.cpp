#include "../include/Target.hpp"

#include <array>
#include <cstdlib>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
// #define TargetDebug
//Debug

std::array<RobotInfo, 5> Robot::Info;

void Robot::Update(const std::vector<ArmorPosi>& armors, double dt)
{
    if(armors.empty()) return;

    if(!is_init)
    {
        this->Init(armors);
        return;
    }

    //更新Robot姿态
    if(armors.size() >= 2)
    {
        this->CorrectAxiRH(armors);
        this->CorrectAttitude();
    }

    #ifdef TargetDebug
    // std::cout<<"anxi: "<<this->axis.x<<" "<<this->axis.y<<" "<<this->axis.z<<"\n";
    // std::cout<<"r_small: "<<this->Info[static_cast<int>(this->type)-1].Rsmall
    //          <<" r_big: "<<this->Info[static_cast<int>(this->type)-1].Rbig
    //          <<" h: "<<this->Info[static_cast<int>(this->type)-1].Hdiff<<"\n";
    #endif
    // 计算装甲板状态
    std::vector< Eigen::Matrix<double, 4, 1> > ArmorStates;
    std::vector<int> indexs;//存储看见的装甲板的索引

    ArmorStates.reserve(armors.size());
    indexs.reserve(armors.size());

    for(auto& armor:armors)
    {
        ArmorStates.emplace_back(armor.posi.x, armor.posi.y, armor.posi.z, this->SolveTheta(armor));
    }

    for(auto& armor : ArmorStates)
    {
        double min_diff = CV_PI; //初始化为180度
        int index;
        for(int i=0;i<4;i++)
        {
            double diff = std::abs(armor(3,0) - this->Armors[i](3,0));
            diff = std::min(diff, 2 * CV_PI - diff); //取最小角度差

            if(diff < min_diff)
            {
                min_diff = diff;
                index = i;
            }
        }
        indexs.emplace_back(index);
    }
    
    //更新装甲板状态
    for(auto& view : this->View)
    {
        view = ArmorView::Invisual;
    }
    for(auto& i : indexs)
    {
        this->View[i] = ArmorView::Visual;
    }

    //kalman滤波更新装甲板状态
    //只有一个装甲板时

    if(armors.size() == 1)
    {
        this->KalmanUpdateOne(ArmorStates[0], dt);
        return;
    }
    //有多个装甲板时
    this->KalmanUpdateTwo(ArmorStates, dt);
}


void Robot::CorrectAxiRH(const std::vector<ArmorPosi>& armors)
{
    //如果输入的装甲板数量不为2，则函数不会执行任何操作
    if(armors.size() != 2) return;
    
    #ifdef TargetDebug
    //观测到的装甲板朝向夹角
    double theta_debug = std::acos((armors[0].toward.dot(armors[1].toward))/(cv::norm(armors[0].toward)*cv::norm(armors[1].toward)));
    theta_debug = (theta_debug/CV_PI)*180.0;
    std::cout <<"armor theta: "<< theta_debug << "\n";
    #endif

    //计算中心轴向量
    cv::Point3d axis_ = armors[0].toward.cross(armors[1].toward);
    axis_ = axis_ / cv::norm(axis_);//单位化

    //统一方向
    double cos0 = axis_.dot(cv::Point3d(0,0,-1));
    if(cos0 < 0) axis_ = -axis_;

    //更新中心轴向量
    this->axis_set.emplace(axis_);
    this->axis_sum += axis_;
    if( this->axis_set.size() >= 5 )
    {
        this->axis_sum -= this->axis_set.back();
        this->axis_set.pop();
    }
    this->axis = this->axis_sum / (double)this->axis_set.size();

//计算半径和高度差
    cv::Point3d armor1_face = axis_.cross(armors[0].toward);
    cv::Point3d armor2_face = axis_.cross(armors[1].toward);
    armor1_face = armor1_face / cv::norm(armor1_face);//单位化
    armor2_face = armor2_face / cv::norm(armor2_face);//单位化

    cv::Point3d armorOneToTwo = armors[1].posi - armors[0].posi;

    double b1 = armorOneToTwo.dot(armor1_face);
    double b2 = armorOneToTwo.dot(armor2_face);

    auto center_ = armors[0].posi + b1 * armor1_face;
    this->center = Eigen::Matrix<double,3,1>{center_.x,center_.y,center_.z};

    double k = armor1_face.dot(armor2_face);
    
    double deno = 1 - k*k;

//计算r
    double r_small_ = ( b1 - b2 * k ) / deno;
    double r_big_ = ( b1 * k - b2 ) / deno;

//判断半径大小
    if(r_small_ > r_big_) std::swap(r_small_,r_big_);

    double h_diff = std::abs(armorOneToTwo.dot(this->axis));

//更新目标装甲板高度差
    #ifdef TargetDebug
    double L_debug = armorOneToTwo.dot(this->axis);
    auto armorOneToTwo_debug = armorOneToTwo - L_debug * this->axis;
    double distance_debug = cv::norm(armorOneToTwo_debug);
    std::cout <<"armor distance: "<< distance_debug << "\n";
    auto armorOneToTwo_debug_unit = armorOneToTwo_debug / distance_debug;
    std::cout<<"anxi: "<<this->axis.x<<" "<<this->axis.y<<" "<<this->axis.z<<"\n";
    std::cout<<"r_small: "<<r_small_
             <<" r_big: "<<r_big_
             <<" h: "<<h_diff
             <<" theta:"<< (std::acos(armor1_face.dot(armorOneToTwo_debug_unit))/CV_PI)*180.0<<"\n";
    #endif

    this->Info[static_cast<int>(this->type)-1](r_small_, r_big_, h_diff);
}

void Robot::CorrectAttitude()
{
    // 1. 定义起始向量 (0, 0, 1)
    cv::Vec3d Z(0, 0, 1);

    // 2. 计算旋转轴 (叉积) 和 旋转角的余弦 (点积)
    // cross product 得到的向量垂直于 source 和 target 构成的平面
    cv::Vec3d rotation_axis = Z.cross(this->axis);
    double dot_prod = Z.dot(this->axis);

    // 3. 处理特殊情况 (共线)
    // 阈值用于处理浮点数误差
    double kEpsilon = 1e-6;

    // 情况 A: 向量已经对齐 (方向相同)
    // 情况 B: 向量反向 (大于90度)，此时叉积为负
    
    if (dot_prod > 1.0 - kEpsilon || dot_prod < 0) { return; }

    // 一般情况C：计算旋转矩阵
    // 使用罗德里格斯变换 (Rodrigues) 
    // OpenCV 的 Rodrigues 函数接受旋转向量 (axis * angle)
    double angle = std::acos(dot_prod); // 此时 dot_prod 在 (-1, 1) 之间
    
    // 归一化旋转轴并乘以角度
    rotation_axis = rotation_axis / cv::norm(rotation_axis);
    cv::Vec3d rotation_vector = rotation_axis * angle;

    cv::Rodrigues(rotation_vector, this->R);
}


double Robot::SolveTheta(const ArmorPosi& armor)
{
    cv::Point3d face = this->axis.cross(armor.toward);
    cv::Matx<double, 3, 1> face_Vec{face.x,face.y,face.z};
    
    face_Vec = this->R * face_Vec;

    return std::atan2(face_Vec(2,0),face_Vec(1,0));
}

void Robot::Init(const std::vector<ArmorPosi>& armors)
{
    if(armors.empty()) return;
    
    this->type = static_cast<Robot::Type>(static_cast<int>(armors[0].type));
    if(armors.size() >= 2)
    {
        this->CorrectAxiRH(armors);
        this->CorrectAttitude();

        this->Armors[0] << armors[0].posi.x, armors[0].posi.y, armors[0].posi.z, this->SolveTheta(armors[0]);
        this->View[0] = ArmorView::Visual;

        this->Armors[1] << armors[1].posi.x, armors[1].posi.y, armors[1].posi.z, this->SolveTheta(armors[1]);
        this->View[1] = ArmorView::Visual;

        double angle_diff = Armors[1](3,0) - Armors[0](3,0);
        
        //调整角度差到[-pi, pi]
        Armors[2](3,0) = Armors[1](3,0) + angle_diff;
        Armors[3](3,0) = Armors[2](3,0) + angle_diff;

        std::remainder(Armors[2](3,0), 2.0 * CV_PI);
        std::remainder(Armors[3](3,0), 2.0 * CV_PI);

        Armors[2].block<3,1>(0,0) = this->Rotate(Armors[1].block<3,1>(0,0), angle_diff);

        Armors[3].block<3,1>(0,0) = this->Rotate(Armors[2].block<3,1>(0,0), angle_diff);

        this->is_init = true;
        return;
    }

    //只有一个装甲板时只初始化该装甲板
    auto face = this->axis.cross(armors[0].toward);
    face = face / cv::norm(face);//单位化

    //初始化中心点
    double& r_small = this->Info[static_cast<int>(this->type)-1].Rsmall;
    auto center_ = armors[0].posi + r_small * face;
    this->center = Eigen::Matrix<double,3,1>{center_.x,center_.y,center_.z};

    this->Armors[0] << armors[0].posi.x, armors[0].posi.y, armors[0].posi.z, this->SolveTheta(armors[0]);
    this->View[0] = ArmorView::Visual;
    
    this->Armors[1](3,0) = this->Armors[0](3,0) + CV_PI/2.0;
    std::remainder(this->Armors[1](3,0), 2.0 * CV_PI);
    this->Armors[1].block<3,1>(0,0) = this->Rotate(this->Armors[0].block<3,1>(0,0), CV_PI/2.0);

    this->Armors[2](3,0) = this->Armors[1](3,0) + CV_PI/2.0;
    std::remainder(this->Armors[2](3,0), 2.0 * CV_PI);
    this->Armors[2].block<3,1>(0,0) = this->Rotate(this->Armors[1].block<3,1>(0,0), CV_PI/2.0);

    this->Armors[3](3,0) = this->Armors[2](3,0) + CV_PI/2.0;
    std::remainder(this->Armors[3](3,0), 2.0 * CV_PI);
    this->Armors[3].block<3,1>(0,0) = this->Rotate(this->Armors[2].block<3,1>(0,0), CV_PI/2.0);

    this->is_init = true;

    return;
}

void Robot::Clear()
{
    this->is_init = false;
    while(!this->axis_set.empty()) this->axis_set.pop();
    this->axis_sum = cv::Point3d(0,0,1);
    this->axis = cv::Point3d(0,0,1);
    this->axis_set.emplace(0,0,1);
}

Eigen::Matrix<double, 3, 1> Robot::Rotate(
    const Eigen::Matrix<double, 3, 1>& Point, 
    double angle )
    {
    // 1. 将 cv::Point3d 转换为 Eigen 类型
    // 这里也可以用 Eigen::Matrix<double,3,1>
    Eigen::Matrix<double, 3, 1> n{this->axis.x, this->axis.y, this->axis.z};

    // 2. 去中心化（平移）
    Eigen::Matrix<double, 3, 1> v = Point - center;

    // 3. 构造旋转
    // AngleAxisd 接受任何 derived from MatrixBase 的 3x1 向量
    Eigen::AngleAxisd rotation_vector{angle, n};

    // 4. 旋转并还原中心
    return (rotation_vector * v) + center;
}



