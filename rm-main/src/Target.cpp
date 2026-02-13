#include "../include/Target.hpp"

#include <array>
#include <cstdlib>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#define TargetDebug
//Debug

std::array<RobotSize, 5> Robot::Size;

void Robot::Update(const std::vector<ArmorPosi>& armors, double dt)
{
    if(armors.empty()) return;

    if(!is_init)
    {
        this->Init(armors);
        return;
    }

    if(armors.size() == 1)
    {
        this->OneArmor(armors[0],dt);
        return;
    }
    else 
    {
        this->TwoArmor(armors,dt);
        return;
    }

}
void Robot::OneArmor(const ArmorPosi& armor, double dt)
{
    // 计算装甲板状态
    Eigen::Matrix<double, 4, 1> ArmorState{armor.posi.x, armor.posi.y, armor.posi.z, this->SolveTheta(armor)};
    
    //装甲板匹配
    int ID = 0; 
    
    double min_diff = CV_PI; //初始化为180度
    for(int i=0;i<4;i++)
    {
        double diff = std::abs(ArmorState(3,0) - this->Armors(3,i));
        diff = std::min(diff, 2 * CV_PI - diff); //取最小角度差

        if(diff < min_diff)
        {
            min_diff = diff;
            ID = i;
        }
    }

    auto& armorView = ArmorState;

    //更新视角
    this->View[(ID+1)%4] = this->View[(ID+2)%4] = this->View[(ID+3)%4] = ArmorView::Invisual;
    this->View[ID] = ArmorView::Visual;

    //卡尔曼滤波
    Eigen::Matrix<double, 8, 1> State;
    State.block<4,1>(0,0) = this->Armors.block<4,1>(0,ID);
    State.block<4,1>(4,0) = this->Speed;

    auto ans = this->Kalman(State, this->CovArmors[ID], armorView, this->Armors(4,ID),dt,armor.error);
    
    //更新Robot信息
    this->Armors.block<4,1>(0,ID) = ans.block<4,1>(0,0);
    this->Speed = ans.block<4,1>(4,0);

    //更新中心点
    double& radius = this->Armors(4,ID);
    double& theta = this->Armors(3,ID);
    Eigen::Matrix<double, 3, 1> face{radius*std::cos(theta), radius*std::sin(theta), 0};

    this->center = this->Armors.block<3,1>(0,ID) - face;

    //更新没有视野的装甲板位置

    //角度
    this->Armors(3,(ID+1)%4) = std::remainder(this->Armors(3,ID)+(CV_PI/2), 2.0 * CV_PI);
    this->Armors(3,(ID+2)%4) = std::remainder(this->Armors(3,ID)+CV_PI, 2.0 * CV_PI);
    this->Armors(3,(ID+3)%4) = std::remainder(this->Armors(3,ID)+(3*CV_PI/2), 2.0 * CV_PI);

    //位置
    double R_ = this->Armors(4,(ID+1)%4);
    double theta_ = this->Armors(3,(ID+1)%4);
    
    this->Armors.block<2,1>(0,(ID+1)%4) = Eigen::Matrix<double,2,1>{this->center(0) + R_*std::cos(theta_), this->center(1) + R_*std::sin(theta_)};

    R_ = this->Armors(4,(ID+2)%4);
    theta_ = this->Armors(3,(ID+2)%4);
    
    this->Armors.block<2,1>(0,(ID+2)%4) = Eigen::Matrix<double,2,1>{this->center(0) + R_*std::cos(theta_), this->center(1) + R_*std::sin(theta_)};    

    R_ = this->Armors(4,(ID+3)%4);
    theta_ = this->Armors(3,(ID+3)%4);
    
    this->Armors.block<2,1>(0,(ID+3)%4) = Eigen::Matrix<double,2,1>{this->center(0) + R_*std::cos(theta_), this->center(1) + R_*std::sin(theta_)};

    //装甲板位置协方差
    this->CovArmors[(ID+1)%4] = this->CovArmors[(ID+2)%4] = this->CovArmors[(ID+3)%4]  = this->Kalman.CovStateInit.block<4,4>(0,0);
    this->CovArmors[ID] = this->Kalman.CovState.block<4,4>(0,0);
}

void Robot::TwoArmor(const std::vector<ArmorPosi>& armors, double dt)
{    //更新Robot姿态

    // 计算装甲板状态
    std::vector< Eigen::Matrix<double, 4, 1> > ArmorStates;
    std::vector<int> IDS;//存储看见的装甲板的索引

    ArmorStates.reserve(armors.size());
    IDS.reserve(armors.size());

    for(auto& armor:armors)
    {
        ArmorStates.emplace_back(armor.posi.x, armor.posi.y, armor.posi.z, this->SolveTheta(armor));
    }

    //装甲板匹配
    for(auto& armor : ArmorStates)
    {
        double min_diff = CV_PI; //初始化为180度
        int index;
        for(int i=0;i<4;i++)
        {
            double diff = std::abs(armor(3,0) - this->Armors(3,i));
            diff = std::min(diff, 2 * CV_PI - diff); //取最小角度差

            if(diff < min_diff)
            {
                min_diff = diff;
                index = i;
            }
        }
        IDS.emplace_back(index);
    }

//匹配完成,更新半径和中心点

    //更新半径
    this->Armors(4,IDS[0]) = Robot::Size[static_cast<int>(this->type)-1].radius[std::min( static_cast<int>(armors[0].radius),1 )];
    this->Armors(4,IDS[1]) = Robot::Size[static_cast<int>(this->type)-1].radius[std::min( static_cast<int>(armors[1].radius),1 )];

    //更新对称半径
    this->Armors(4,(IDS[0]+2)%4) = this->Armors(4,IDS[0]);
    this->Armors(4,(IDS[1]+2)%4) = this->Armors(4,IDS[1]);

    //更新中心点
    for(int i=0; i<4; i++)
    {
        if(this->View[i] == ArmorView::Invisual) continue;

        double& theta = this->Armors(3,i);
        double& radius = this->Armors(4,i);
        Eigen::Matrix<double, 3, 1> face{radius*std::cos(theta), radius*std::sin(theta), 0};        

        this->center = this->Armors.block<3,1>(0,i) - face;
        break;
    }

//采用最优选取一个装甲板用卡尔曼估计速度
    int index = 0;
    double minAngle = CV_PI/2.0;

    for(int i=0; i<2; i++)
    {
        if(std::abs(armors[i].theta) < minAngle)
        {
            minAngle = std::abs(armors[i].theta);
            index = i;
        }
    }
    
    //拿到要用卡尔曼滤波器更新的装甲板的ID
    int ID = IDS[index];
    auto& armorView = ArmorStates[index];

    //更新视角
    this->View[(ID+1)%4] = this->View[(ID+2)%4] = this->View[(ID+3)%4] = ArmorView::Invisual;
    this->View[ID] = ArmorView::Visual;

    //卡尔曼滤波
    Eigen::Matrix<double, 8, 1> State;
    State.block<4,1>(0,0) = this->Armors.block<4,1>(0,ID);
    State.block<4,1>(4,0) = this->Speed;

    auto ans = this->Kalman(State, this->CovArmors[ID], armorView, this->Armors(4,ID),dt,armors[index].error);
    
    //更新Robot信息
    this->Armors.block<4,1>(0,ID) = ans.block<4,1>(0,0);
    this->Speed = ans.block<4,1>(4,0);

    //更新中心点
    double& radius = this->Armors(4,ID);
    double& theta = this->Armors(3,ID);
    Eigen::Matrix<double, 3, 1> face{radius*std::cos(theta), radius*std::sin(theta), 0};

    this->center = this->Armors.block<3,1>(0,ID) - face;
    
    //更新其他装甲板的位置
    int NoChooseIndex = 1 - index;
    this->Armors.block<4,1>(0,IDS[NoChooseIndex]) = ArmorStates[NoChooseIndex];

    //更新没有视野的装甲板位置

    //角度
    this->Armors(3,(ID+2)%4) = std::remainder(this->Armors(3,ID)+CV_PI, 2.0 * CV_PI);
    this->Armors(3,(IDS[NoChooseIndex]+2)%4) = std::remainder(this->Armors(3,IDS[NoChooseIndex])+CV_PI, 2.0 * CV_PI);

    //位置
    this->Armors.block<3,1>(0,(ID+2)%4) = this->Rotate(this->Armors.block<3,1>(0,ID), CV_PI);
    this->Armors.block<3,1>(0,(IDS[NoChooseIndex]+2)%4) = this->Rotate(this->Armors.block<3,1>(0,IDS[NoChooseIndex]), CV_PI);

    //装甲板位置协方差
    this->CovArmors[ID] = this->Kalman.CovState.block<4,4>(0,0);
    this->CovArmors[IDS[NoChooseIndex]] = this->Kalman.CovView;
    this->CovArmors[(ID+2)%4] = this->Kalman.CovStateInit.block<4,4>(0,0);
    this->CovArmors[(IDS[NoChooseIndex]+2)%4] = this->Kalman.CovStateInit.block<4,4>(0,0);

}


/**
 * @brief 如果输入的装甲板数量为2，函数将更新Robot的尺寸
 * @param armors 输入的装甲板位置，必须包含两个装甲板
 * @note 如果输入的装甲板数量不为2，函数不会执行任何操作
 */
 void Robot::SolveRobotSize(std::vector<ArmorPosi>& armors)
{
    //如果输入的装甲板数量不为2，则函数不会执行任何操作
    if(armors.size() != 2 || armors[0].type != armors[1].type) return;
    
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
    double cos0 = axis_.dot(cv::Point3d(0,-1,0));
    if(cos0 < 0) axis_ = -axis_;


//计算半径和高度差
    cv::Point3d armor1_face = axis_.cross(armors[0].toward);
    cv::Point3d armor2_face = axis_.cross(armors[1].toward);
    armor1_face = armor1_face / cv::norm(armor1_face);//单位化
    armor2_face = armor2_face / cv::norm(armor2_face);//单位化

    cv::Point3d armorOneToTwo = armors[1].posi - armors[0].posi;

    double b1 = armorOneToTwo.dot(armor1_face);
    double b2 = armorOneToTwo.dot(armor2_face);


    double k = armor1_face.dot(armor2_face);
    
    double deno = 1 - k*k;

//计算r
    double ArmorOneR = std::abs( ( b1 - b2 * k ) / deno );
    double ArmorTwoR = std::abs( ( b1 * k - b2 ) / deno );

//判断半径大小,并更新Robot尺寸
    if(ArmorOneR < ArmorTwoR) 
    {
        armors[0].radius = ArmorPosi::Radius::Short;
        armors[1].radius = ArmorPosi::Radius::Long;
        Robot::Size[static_cast<int>(armors[0].type)-1](ArmorOneR, ArmorTwoR);
    }
    else
    {
        armors[0].radius = ArmorPosi::Radius::Long;
        armors[1].radius = ArmorPosi::Radius::Short;
        Robot::Size[static_cast<int>(armors[0].type)-1](ArmorTwoR, ArmorOneR);
    }
}





double Robot::SolveTheta(const ArmorPosi& armor)
{
    auto p = armor.toward.cross(cv::Point3d{0,0,1});
    return std::atan2(p.y,p.x);
}

void Robot::Init(const std::vector<ArmorPosi>& armors)
{
    if(armors.empty()) return;
    
    //装甲板类型检查
    if(static_cast<int>(armors[0].type) < 1 || static_cast<int>(armors[0].type) > 5) return;

    this->type = static_cast<Robot::Type>(static_cast<int>(armors[0].type));
    this->Kalman.Init();

    int ID = 0;
    double minAngle = CV_PI/2.0;

    for(int i=0; i<armors.size(); i++)
    {
        if(std::abs(armors[i].theta) < minAngle)
        {
            minAngle = std::abs(armors[i].theta);
            ID = i;
        }
    }

//初始化中心点
    double& radius = Robot::Size[static_cast<int>(this->type)-1].radius[std::min( static_cast<int>(armors[ID].radius),1 )];

    auto face = cv::Point3d(this->axis(0,0), this->axis(1,0), this->axis(2,0)).cross(armors[ID].toward);
    face = face / cv::norm(face);
    this->center = Eigen::Matrix<double,3,1>{armors[ID].posi.x + radius * face.x,
                                             armors[ID].posi.y + radius * face.y,
                                             armors[ID].posi.z + radius * face.z};

//有两个装甲板时初始化所有装甲板
    if(armors.size() >= 2)
    {

        //初始化能看见的装甲板位置信息（x,y,z,theta）
        this->Armors.col(0) << armors[0].posi.x, armors[0].posi.y, armors[0].posi.z, this->SolveTheta(armors[0]);
        this->View[0] = ArmorView::Visual;

        this->Armors.col(1) << armors[1].posi.x, armors[1].posi.y, armors[1].posi.z, this->SolveTheta(armors[1]);
        this->View[1] = ArmorView::Visual;

        //初始化每个装甲板的半径信息(radius)
        if(armors[0].radius == ArmorPosi::Radius::Short)
        {
            this->Armors(4,0) = this->Armors(4,2) =  Robot::Size[static_cast<int>(this->type)-1].radius[0];
            this->Armors(4,1) = this->Armors(4,3) =  Robot::Size[static_cast<int>(this->type)-1].radius[1];
        }
        else
        {
            this->Armors(4,1) = this->Armors(4,3) =  Robot::Size[static_cast<int>(this->type)-1].radius[0];
            this->Armors(4,0) = this->Armors(4,2) =  Robot::Size[static_cast<int>(this->type)-1].radius[1];
        }
        

//初始化没看见的装甲板位置信息（x,y,z,theta）

        //初始化角度
        //调整角度差到[-pi, pi]
        this-> Armors(3,2) = std::remainder(this->Armors(3,0)+CV_PI, 2.0 * CV_PI);
        this-> Armors(3,3) = std::remainder(this->Armors(3,1)+CV_PI, 2.0 * CV_PI);

        //初始化没看见的装甲板位置x,y,z
        this->Armors.block<3,1>(0,2) = this->Rotate(this->Armors.block<3,1>(0,0), CV_PI);
        this->Armors.block<3,1>(0,3) = this->Rotate(this->Armors.block<3,1>(0,1), CV_PI);

        //初始化装甲板协方差
        this->CovArmors[0] = this->CovArmors[1] = this->Kalman.CovView;
        this->CovArmors[2] = this->CovArmors[3] = this->Kalman.CovStateInit.block<4,4>(0,0);

        this->is_init = true;
        return;
    }

//只有一个装甲板时只初始化该装甲板

    //初始化能看见的装甲板位置信息（x,y,z,theta）
    this->Armors.col(0) << armors[0].posi.x, armors[0].posi.y, armors[0].posi.z, this->SolveTheta(armors[0]);
    this->View[0] = ArmorView::Visual;
    
    //初始化每个装甲板的半径信息(radius)
    this->Armors.row(4) = Eigen::Matrix<double, 1, 4>{ radius, radius, radius, radius };

//初始化没看见的装甲板位置信息（x,y,z,theta）

    //初始化角度

    //调整角度差到[-pi, pi]
    this-> Armors(3,1) = std::remainder(this->Armors(3,0) + (CV_PI/2),   2.0 * CV_PI);
    this-> Armors(3,2) = std::remainder(this->Armors(3,0) +  CV_PI,       2.0 * CV_PI);
    this-> Armors(3,3) = std::remainder(this->Armors(3,0) + (CV_PI*3/2), 2.0 * CV_PI);

    //初始化没看见的装甲板位置x,y,z
    this->Armors.block<3,1>(0,1) = this->Rotate(this->Armors.block<3,1>(0,0), CV_PI/2);
    this->Armors.block<3,1>(0,2) = this->Rotate(this->Armors.block<3,1>(0,0), CV_PI);
    this->Armors.block<3,1>(0,3) = this->Rotate(this->Armors.block<3,1>(0,0), CV_PI*3/2);

    //初始化装甲板协方差
    this->CovArmors[0] = this->Kalman.CovView;
    this->CovArmors[1] = this->CovArmors[2] = this->CovArmors[3] = this->Kalman.CovStateInit.block<4,4>(0,0);    
    
    this->is_init = true;

    return;
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



