#ifndef LIGHT_AND_ARMOR_STRUCT
#define LIGHT_AND_ARMOR_STRUCT
#include <array>
#include <cstdlib>
#include <eigen3/Eigen/Core>
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>
struct Light{
    explicit Light(const cv::RotatedRect& rect)
    {
        cv::Point2f p[4];  //灯条四个点
        rect.points(p);  //rotatedrect的points函数可以获取四个点的坐标
        std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
        
        top = (p[0] + p[1]) / 2;//灯条顶部和底部中心点
        bottom = (p[2] + p[3]) / 2;

        this->center = rect.center;
        this->width = cv::norm(p[1]-p[0]);
        this->length = cv::norm(top - bottom);

    }

    cv::Point2f top, bottom;
    cv::Point2f center;

    double length;
    double width;
    enum class Color : int { Red = 0, Blue = 1 } color;
};



struct CVArmor{
    CVArmor(const Light& light1, const Light& light2):
        left(light1), right(light2)
    {
        if(light1.center.x>light2.center.x) std::swap(left, right);
        Lightcorners.resize(4);
        Lightcorners[0] = left.top;
        Lightcorners[1] = right.top;
        Lightcorners[2] = right.bottom;
        Lightcorners[3] = left.bottom;
    }
    Light left, right;
    std::vector<cv::Point2f> Lightcorners; //装甲板四个顶点
};

struct ArmorPosi{
    //顺时针yaw和逆时针yaw
    Eigen::Matrix<double, 3, 2> center;
    Eigen::Vector3d photocenter;
    std::array<double, 2> yaw;
    std::array<double, 2> reproj;
    std::array<double, 2> theta;//装甲板中心点方向角
    std::array<double, 2> yaw_abs;
    Eigen::Matrix<double, 3, 2> SCS;    //球坐标系坐标，以相机系为坐标系

    bool IsInRange; // 是否在有效范围内的标志位

    enum class Type : int {base    = 0, hero     = 1, two   = 2,
                           three   = 3, four     = 4, guard = 5,
                           outpost = 6, Unknow = 7} type = Type::Unknow;
    float confidence = 0; 

    ArmorPosi(const Eigen::Matrix<double, 3, 2>& center,
              const Eigen::Matrix<double, 3, 2>& center_cam, 
              const Eigen::Vector3d& photocenter,
              std::array<double, 2> yaw, 
              std::array<double, 2> reproj,
              bool isInRange):
              center(center), photocenter(photocenter), yaw(yaw), reproj(reproj), IsInRange(isInRange){

                this->theta[0] = std::atan2(this->center(1, 0), this->center(0, 0));
                this->theta[1] = std::atan2(this->center(1, 1), this->center(0, 1));

                this->yaw_abs[0] = std::abs( std::remainder(this->yaw[0] + M_PI - this->theta[0], M_PI*2.0) );
                this->yaw_abs[1] = std::abs( std::remainder(this->yaw[1] + M_PI - this->theta[1], M_PI*2.0) );

                this->SCS(0,0) = center_cam.col(0).norm();
                double distance0 = center_cam.col(0).block<2,1>(0,0).norm();
                this->SCS(1,0) = std::asin(distance0 / this->SCS(0,0));
                this->SCS(2,0) = std::atan2(center_cam(1,0), center_cam(0,0));

                this->SCS(0,1) = center_cam.col(1).norm();
                double distance1 = center_cam.col(1).block<2,1>(0,0).norm();
                this->SCS(1,1) = std::asin(distance1 / this->SCS(0,1));
                this->SCS(2,1) = std::atan2(center_cam(1,1), center_cam(0,1));
              }

    ArmorPosi():IsInRange(false){}
    };
struct YoloArmor{ 
    cv::Rect box;
    float conf;
    int class_id;
    std::vector<cv::Point2f> keypoints;
};
#endif
