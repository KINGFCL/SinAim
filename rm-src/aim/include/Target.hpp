#ifndef TARGET_HPP
#define TARGET_HPP

#include "Armor.hpp"
#include "EKFKalman.hpp"
#include "LKFKalman.hpp"

#include <chrono>
#include <cstddef>
#include <deque>
#include <eigen3/Eigen/Core>
#include <array>
#include <opencv2/core/types.hpp>
#include <queue>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

struct RobotSize
{
//目标半径,0小半径,1大半径
    std::array<double,2> radius{20.0,30.0}; 

    RobotSize() = default;

    void operator () (double r_small, double r_big)
    {
        radius[0] = r_small;
        radius[1] = r_big;
    }
    
};

class Robot
{
public:
    enum class KalmanMode : int
    {
        LKF = 0,
        EKF = 1
    };
    Robot() = default;

    enum class ArmorView : bool
    {
        Visual = true,
        Invisual = false
    };

    enum class Type : int
    {hero  = 1, two  = 2,
     three = 3, four = 4, 
     guard = 5} type;   
    

    /*!
    看见两个装甲板，更新Robot的尺寸
    @param armors 输入的装甲板位置，必须包含一个或者两个装甲板
    /注意：如果只传如一个装甲板，则不会更新尺寸
    */
    static void SolveRobotSize(std::vector<ArmorPosi>& armors);

    /*!
    第一次看见Robot，初始化Robot的状态
    @param armors 输入的装甲板位置，必须包含一个或者两个装甲板
    */
    void Init(const std::vector<ArmorPosi>& armors);
    void InitEKF(const Eigen::Vector3d& center, const Eigen::Vector3d& SCS, double yaw);
    void InitLKF(const Eigen::Vector3d& center, const Eigen::Vector3d& SCS);

    KalmanMode GetMode() const { return this->Mode; }

    /*!
    完全丢失Robot，清空。
    /回到未初始化状态
    */
    void Clear();

    /*!
    新数据到来，更新Robot的状态
    @param armors 输入的装甲板位置，必须包含一个或者两个装甲板
    /装甲板数量大于2时只会使用前两个装甲板进行更新，装甲板为空不做任何操作
    */
    void Update(const std::vector<ArmorPosi>& armors,const Eigen::Quaterniond& gripper_to_world,double dt);
    void Update(double dt);

    void UpdateLKF(const Eigen::Vector3d& armorcenter, const Eigen::Vector3d& SCS, double dt);//这里得改
    // void UpdateLKF(const Eigen::Vector3d& armorcenter, const Eigen::Vector3d& SCS,const Eigen::Quaterniond& gripper_to_world, double dt);
    void UpdateLKF(double dt);

    void UpdateEKF(const std::vector<ArmorPosi>& armors,const Eigen::Quaterniond& gripper_to_world,double dt);
    void UpdateEKF(double dt);

    void OneArmor(const ArmorPosi& armor,const Eigen::Quaterniond& gripper_to_world, double dt);
    void TwoArmor(const std::vector<ArmorPosi>& armors,const Eigen::Quaterniond& gripper_to_world, double dt);

    /*!
    @return 返回当前装甲板的朝向角
    */
    double SolveTheta(const ArmorPosi& armors) ;

    Eigen::Matrix<double,4,4> Predict(double dt) ;//mm/s, rad/s

    Eigen::Matrix<double,4,4> Predict(double dt, Eigen::Vector3d& center);

    /*!
    @return 旋转点Point绕轴axis旋转angle角度后的新坐标
    */
    Eigen::Matrix<double, 3, 1> Rotate( const Eigen::Matrix<double, 3, 1>& center, double angle );

public:
//整车速度v_x,v_y,v_z,w
    Eigen::Matrix<double, 4, 1> Speed;

/* 4个装甲板位置：
    ID:   0        1          2          3

    0     x        x          x          x
    1     y        y          y          y
    2     z        z          z          z
    3   theta    theta      theta      theta
    4   radius   radius     radius     radius
*/

/*
    theta radius h 
*/
    Eigen::Matrix<double, 3, 4> Armors;
    double l_diff = 0, h_diff = 0;
    const double r = 24.0;
    double d_theta_1 = CV_PI/2, d_theta_2 = CV_PI, d_theta_3 = -CV_PI/2;

    std::array<ArmorView, 4> View = {ArmorView::Invisual, ArmorView::Invisual, ArmorView::Invisual, ArmorView::Invisual};


    //中心转轴方向
    const Eigen::Matrix<double,3,1> axis{0,0,1};//单位向量
    
    //中心点坐标
    Eigen::Matrix<double,3,1> center{0,0,0};

    EKFKalman ekfkalman;
    LKFKalman lkfkalman;

private:
    struct MatchAns
    {
        size_t id;
        size_t side;
        double err;
    };

    double MatchErrorInLKF(const ArmorPosi& armor, double dt);

    MatchAns MatchErrorInEKF(const ArmorPosi& armor, double dt);
    std::pair< MatchAns, MatchAns> MatchErrorInEKF(const std::vector<ArmorPosi>& armors,double dt);

    void LKFToEKF(const std::vector<ArmorPosi>& armors);
    void LKFToEKF(const ArmorPosi& armor, size_t side);

    KalmanMode Mode = KalmanMode::LKF;

    const double matcherrthresh = 100.0;//单位：cm 

    //记录是否初始化
    bool is_init = false;
    
    // static std::array<RobotSize, 5> Size;//不同类型机器人信息
};




#endif // TARGET_HPP