#ifndef ARMOR_DETECTOR__RESNET_NUMBER_CLASSIFIER_HPP_
#define ARMOR_DETECTOR__RESNET_NUMBER_CLASSIFIER_HPP_

#include "Armor.hpp"
#include <array>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

class ResNetNumClassifier
{
public:

    /**
    * @brief 神经网络模型输出 ID 与装甲板类别的映射表
    * * | ID (label_id) | 枚举成员 (ArmorName) | 对应机器人 / 目标类型       |
    * | :------------ | :------------------ | :------------------------ |
    * | 0             | one                 | 1号 (英雄)                |
    * | 1             | two                 | 2号 (工程)                |
    * | 2             | three               | 3号 (步兵)                |
    * | 3             | four                | 4号 (步兵)                |
    * | 4             | five                | 5号 (步兵)                |
    * | 5             | sentry              | 哨兵 (Sentry)             |
    * | 6             | outpost             | 前哨站 (Outpost)          |
    * | 7             | base                | 基地 (Base)               |
    * | 8             | not_armor           | 非装甲板 (背景/误检)       |
    */
    struct Ans {
        int id;
        float confidence;
        Ans() : id(0), confidence(0.0f) {}
        Ans(int id, float con) : id(id), confidence(con) {}
    };

    explicit ResNetNumClassifier(std::string model_path, float confidence_threshold = 0.5f);

    std::vector<ArmorPosi> operator()(std::vector<std::array<ArmorPosi, 2>>& armors,
                                      const std::vector<cv::Mat>& armors_pattern);

    std::vector<Ans> Classify(const std::vector<std::array<ArmorPosi, 2>>& armors, const std::vector<cv::Mat>& armors_pattern);

private:
    float confidence_threshold;
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    static constexpr size_t MAX_BATCH = 4;
};

#endif
