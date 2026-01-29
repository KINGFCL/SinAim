#ifndef TARGET_HPP
#define TARGET_HPP

#include "Armor.hpp"

#include <deque>
#include <eigen3/Eigen/Core>
#include <array>
#include <opencv2/core/types.hpp>
#include <queue>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

struct RobotInfo
{
//目标半径
    double Rsmall, Rbig; //目标半径mm

    double RSmallSum = 0.0;//目标半径和
    double RBigSum = 0.0;//目标半径和

//装甲板沿转轴方向的高度差,小半径到大半径的高度差
    double Hdiff;
    double HSum = 0;

//观测次数
    int count = 0;
    RobotInfo(double r_small = 200.0,double r_big = 300.0,double h_diff = 0.0)
        : Rsmall(r_small), Rbig(r_big), Hdiff(h_diff) {}

    void operator () (double r_small, double r_big, double h_diff)
    {
        this->RSmallSum += r_small;
        this->RBigSum += r_big;
        this->HSum += h_diff;
        this->count++;
        this->Rsmall = RSmallSum / (double)count;
        this->Rbig = RBigSum / (double)count;
        this->Hdiff = HSum / (double)count;
    }
    
};

class Robot
{
public:
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
    第一次看见Robot，初始化Robot的状态
    @param armors 输入的装甲板位置，必须包含一个或者两个装甲板
    */
    void Init(const std::vector<ArmorPosi>& armors);

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
    void Update(const std::vector<ArmorPosi>& armors, double dt);

    /*!
    修正Robot的转轴方向，旋转半径，高度差
    @param armors 输入的装甲板位置，必须包含两个装甲板
    \ 警告：如果输入的装甲板数量不为2，则函数不会执行任何操作
    */
    void CorrectAxiRH(const std::vector<ArmorPosi>& armors);

    /*!
    更新Robot的姿态信息
    */
    void CorrectAttitude();

    /*!
    @return 返回当前装甲板的朝向角
    */
    double SolveTheta(const ArmorPosi& armors) ;

    Eigen::Matrix<double,8,1> Predict(const Eigen::Matrix<double,8,1>& X,double dt) ;//mm/s, rad/s

    void KalmanUpdateOne(const Eigen::Matrix<double,4,1>& measure,double dt);
    void KalmanUpdateTwo(const std::vector<Eigen::Matrix<double,4,1>>& measures,double dt);


    //std::vector< Eigen::Matrix<double,4,1> > EveryArmorState(const std::vector< Eigen::Matrix<double,4,1> >& X) ;

    /*!
    @return 旋转点Point绕轴axis旋转angle角度后的新坐标
    */
    Eigen::Matrix<double, 3, 1> Rotate( const Eigen::Matrix<double, 3, 1>& Point, double angle );
public:
//整车状态
    Eigen::Matrix<double, 8, 1> State; //x,y,z,theta,v_x,v_y,v_z,w
    //4个装甲板位置
    std::array<Eigen::Matrix<double, 4, 1>, 4> Armors;
    std::array<ArmorView, 4> View = {ArmorView::Invisual, ArmorView::Invisual, ArmorView::Invisual, ArmorView::Invisual};

    //记录是否初始化
    bool is_init = false;


    //中心转轴方向
    cv::Point3d axis = cv::Point3d(0,0,1);//单位向量
    
    //中心点坐标
    Eigen::Matrix<double,3,1> center{0,0,0};

    //坐标系的变化矩阵
    cv::Matx<double, 3, 3> R{1,0,0,
                             0,1,0,
                             0,0,1};


//中心转轴
    std::queue<cv::Point3d> axis_set{std::deque<cv::Point3d>{cv::Point3d(0,0,1)}};
    cv::Point3d axis_sum{0,0,1};



private:
    static std::array<RobotInfo, 5> Info;//不同类型机器人信息
};



// class Target
// {
// public:
//     virtual Eigen::Matrix<double,8,1> Predict(const Eigen::Matrix<double,8,1>& X,double dt) = 0;
//     virtual std::vector< Eigen::Matrix<double,8,1> > EveryArmorState(const std::vector< Eigen::Matrix<double,8,1> >& X) = 0;
    
//     /*!
//     @return 返回当前装甲板的朝向角
//     */
//     virtual double SolveAngel(const ArmorPosi& armors) = 0;

//     //中心转轴方向
//     cv::Point3d axis = cv::Point3d{0,0,1};//单位向量
    
//     //中心点坐标
//     Eigen::Matrix<double,3,1> center{0,0,0};

//     //对象速度向量
//     Eigen::Matrix<double,4,1> speed{0,0,0,0};
// };
#endif // TARGET_HPP