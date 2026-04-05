#include "MlpNumClassifier.hpp"
#include <opencv2/core/types.hpp>
#include <vector>


MlpNumClassifier::MlpNumClassifier(std::string model_path, float confidence_threshold)
    : confidence_threshold(confidence_threshold)
{
    Net = cv::dnn::readNetFromONNX(model_path);
    // 设置首选的计算后端为 OpenVINO Inference Engine
    // Net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    // 设置首选的计算目标设备为 CPU
    Net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    // Create blob from image

}
std::vector<MlpNumClassifier::Ans> MlpNumClassifier::Classify(const std::vector<cv::Mat>& armors_pattern)
{
    std::vector<MlpNumClassifier::Ans> ans;
    if(armors_pattern.empty()) return ans;
    ans.reserve(armors_pattern.size());

    // Create blob from image
    cv::Mat blob;
    cv::dnn::blobFromImages(armors_pattern, blob, 1/255.0);

     // Set the input blob for the neural network
    this->Net.setInput(blob);
    // Forward pass the image blob through the model
    cv::Mat outputs = this->Net.forward();

    //读取结果
    for (int i = 0; i < outputs.rows; ++i) 
    {
        // 获取第 i 张图片对应的得分行
        cv::Mat scores = outputs.row(i);

        // Do softmax
        float max_prob = *std::max_element(scores.begin<float>(), scores.end<float>()); //max_element函数返回的是一个迭代器，要获取实际值，需要解引用*
        cv::Mat softmax_prob;
        cv::exp(scores - max_prob, softmax_prob);
        float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
        softmax_prob /= sum;

        double confidence;
        cv::Point class_id_point;
        minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
        int label_id = class_id_point.x;
        ans.emplace_back(label_id,confidence);
    }
    return ans;
}
std::vector<ArmorPosi> MlpNumClassifier::operator()(std::vector< std::array<ArmorPosi,2> >& armors,const std::vector<cv::Mat>& armors_pattern)
{
    std::vector<ArmorPosi> result;
    if(armors.empty()) return result;

    result.reserve(armors.size());

    std::vector<MlpNumClassifier::Ans> ans = Classify(armors_pattern);
    
    for (size_t i = 0; i < ans.size(); i++) {
        // 1. 置信度过滤
        if (ans[i].confidence > this->confidence_threshold) {
            int model_id = ans[i].id; // 模型的原始 ID (0-8)
            ArmorPosi::Type final_type = ArmorPosi::Type::Unknow;
            int target_idx = 0; // 默认小装甲板

            // 2. 映射逻辑：将模型 ID (ArmorName 顺序) 映射到你的 Type 枚举
            switch (model_id) {
                case 0: // 模型 0 = 英雄 (one)
                    final_type = ArmorPosi::Type::hero;
                    target_idx = 1; // 英雄是大装甲板
                    break;
                case 1: // 模型 1 = 工程 (two)
                    final_type = ArmorPosi::Type::two;
                    target_idx = 0; // 工程通常是小装甲板
                    break;
                case 2: // 模型 2 = 3号步兵 (three)
                    final_type = ArmorPosi::Type::three;
                    target_idx = 0;
                    break;
                case 3: // 模型 3 = 4号步兵 (four)
                    final_type = ArmorPosi::Type::four;
                    target_idx = 0;
                    break;
                case 5: // 模型 5 = 前哨站 (outpost)
                    final_type = ArmorPosi::Type::outpost;
                    target_idx = 0;
                    break;
                case 6: // 模型 6 = 哨兵 (sentry)
                    final_type = ArmorPosi::Type::guard;
                    target_idx = 0;
                    break;
                case 7: // 模型 7 = 基地 (base)
                    final_type = ArmorPosi::Type::base;
                    target_idx = 1; // 基地是大装甲板
                    break;
                case 4: // 5号步兵已不存在，跳过或设为未知
                case 8: // 非装甲板
                default:
                    final_type = ArmorPosi::Type::Unknow;
                    break;
            }

            // 3. 过滤掉无效目标
            if (final_type == ArmorPosi::Type::Unknow) continue;

            // 4. 压入结果
            result.emplace_back(armors[i][target_idx]);
            result.back().type = final_type;
            result.back().confidence = ans[i].confidence;
        }
    }
    
    return result;
}