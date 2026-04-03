#ifndef __INCLUDE_CVDETECTOR_CLASS__
#define __INCLUDE_CVDETECTOR_CLASS__
#include "Armor.hpp"
#include <deque>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
class CVDetector {
public:
    CVDetector(Light::Color color,float confidence);
    
    std::deque<CVArmor> operator () (cv::Mat& frame,std::vector<cv::Mat>& armors_pattern);
    std::deque<CVArmor> operator () (cv::Mat& frame,std::vector<cv::Mat>& armors_pattern,bool isSmallROI);

    void ArmorShow(cv::Mat & rgb_img, const std::deque<CVArmor> & armors);
    void ArmorShow(cv::Mat & rgb_img, const std::vector<CVArmor> & armors);

public:
    float confidence;
    Light::Color color;
    cv::Mat gray_img;
    cv::Mat rgb_img;
    
public:
    cv::Mat preprocessImage(cv::Mat& rgb_img); //图像预处理
    std::deque<Light> FindLight(const cv::Mat & binary_img); //寻找灯条
    std::deque<CVArmor> FindArmor(const std::deque<Light> & lights); //寻找装甲板
    std::vector<cv::Mat> ROIArmor(const std::deque<CVArmor>& armors);
    std::vector<cv::Mat> SmallROIArmor(const std::deque<CVArmor> & armors);
};
#endif