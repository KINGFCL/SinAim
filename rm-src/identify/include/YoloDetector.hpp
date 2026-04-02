#ifndef YOLO11_DETECTOR_HPP
#define YOLO11_DETECTOR_HPP
#include "Armor.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

class YOLO11Detector {

public:
    enum class Camp : bool { Blue = false, Red = true };

public:
    // 构造函数：加载模型并配置 PrePostProcessor
    YOLO11Detector(const std::string& model_path, Camp camp, float conf_threshold = 0.5f, float nms_threshold = 0.2f, const std::string& device = "AUTO");

    // 执行推理和后处理
    std::vector<YoloArmor> operator()(const cv::Mat& raw_img);

    // 可视化函数
    void draw(cv::Mat& img, const std::vector<YoloArmor>& detections) const;

private:
    // Letterbox 图像预处理
    cv::Mat letterbox(const cv::Mat& source, float& scale, int& pad_w, int& pad_h) const;
    
    Camp camp_;
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    cv::Mat raw_output_;        // 用于接收 OpenVINO 的原始输出 [50, 8400]
    cv::Mat transposed_output_; // 用于存放转置后的结果 [8400, 50]

    float conf_threshold_;
    float nms_threshold_;
    int input_width_ = 640;
    int input_height_ = 640;
    int class_num_ = 38; // 你的模型有 38 个分类

    const std::vector<std::string> class_names_ = {
        "Bsentry", "Rsentry", "Esentry", "Bone", "Rone", "Eone", "Btwo", "Rtwo", "Etwo", 
        "Bthree", "Rthree", "Ethree", "Bfour", "Rfour", "Efour", "Bfive", "Rfive", "Efive", 
        "Boutpost", "Routpost", "Eoutpost", "Bbase", "Rbase", "Ebase", "Pbase", 
        "Bbasesmall", "Rbasesmall", "Ebasesmall", "Pbasesmall", 
        "Bbalancethree", "Rbalancethree", "Ebalancethree", 
        "Bbalancefour", "Rbalancefour", "Ebalancefour", 
        "Bbalancefive", "Rbalancefive", "Ebalancefive"
    };
};

#endif