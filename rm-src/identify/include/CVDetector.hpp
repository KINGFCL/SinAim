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
    enum class ROIType : int { ResNet = 0, MLP = 1 };

    struct CVDetectorConfig {
        int color;                // Red = 0, Blue = 1
        int roi_width  = 32;
        int roi_height = 32;
    };

public:
    explicit CVDetector(Light::Color color, cv::Size ROISize = cv::Size(32, 32));
    explicit CVDetector(const CVDetectorConfig& config);

    
    std::vector<CVArmor> operator () (cv::Mat& frame, std::vector<cv::Mat>& armors_pattern);
    std::vector<CVArmor> operator () (cv::Mat& frame, std::vector<cv::Mat>& armors_pattern, ROIType Type);

    void ArmorShow(cv::Mat & rgb_img, const std::deque<CVArmor> & armors);
    void ArmorShow(cv::Mat & rgb_img, const std::vector<CVArmor> & armors);

private:

    const Light::Color color;
    const cv::Size ROISize;

public:

    cv::Mat gray_img;
    cv::Mat rgb_img;

    
public:
    cv::Mat preprocessImage(cv::Mat& rgb_img); //图像预处理
    std::deque<Light> FindLight(const cv::Mat & binary_img); //寻找灯条
    std::vector<CVArmor> FindArmor(const std::deque<Light> & lights); //寻找装甲板
    std::vector<cv::Mat> ResNetROIPattern(const std::vector<CVArmor>& armors);
    std::vector<cv::Mat> MlpROIPattern(const std::vector<CVArmor> & armors);
};
#endif