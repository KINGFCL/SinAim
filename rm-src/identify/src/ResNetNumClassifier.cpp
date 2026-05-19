#include "ResNetNumClassifier.hpp"
#include "Armor.hpp"
#include <cmath>
#include <cstring>
#include <iostream>

ResNetNumClassifier::ResNetNumClassifier(std::string model_path, float confidence_threshold)
    : confidence_threshold(confidence_threshold)
{
    auto model_base = core.read_model(model_path);

    // 预处理融合：让模型直接接收 uint8 灰度图，内部自动归一化到 [0, 1]
    ov::preprocess::PrePostProcessor ppp(model_base);
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NCHW"); // [Batch, Channel=1, H=32, W=32]
    ppp.input().preprocess()
        .convert_element_type(ov::element::f32)
        .scale(255.0f);
    auto model = ppp.build();

    // 固定 batch 大小为 MAX_BATCH，避免动态 shape 带来的额外开销
    std::map<std::string, ov::PartialShape> shapes;
    shapes[model->input().get_any_name()] = ov::PartialShape{MAX_BATCH, 1, 32, 32};
    model->reshape(shapes);

    // 编译到 GPU，LATENCY 模式优先降低单次推理延迟
    ov::AnyMap props;
    props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    props[ov::cache_dir.name()] = "./gpu_cache"; // 缓存编译结果，加快下次启动
    compiled_model = core.compile_model(model, "CPU", props);
    infer_request = compiled_model.create_infer_request();

    std::cout << "[ResNetNumClassifier] GPU loaded\n";
}

std::vector<ResNetNumClassifier::Ans> ResNetNumClassifier::Classify(const std::vector<std::array<ArmorPosi, 2>>& armors,const std::vector<cv::Mat>& armors_pattern)
{
    size_t N = armors_pattern.size();
    std::vector<Ans> ans(N);
    if (N == 0) return ans;

    // 先验过滤：两个解算结果都不在有效范围内，直接判为非装甲板，跳过推理
    std::vector<size_t> need_infer;
    need_infer.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        if (!armors[i][0].IsInRange && !armors[i][1].IsInRange) {
            ans[i] = Ans(8, 1.0f);
        } else {
            need_infer.push_back(i);
        }
    }

    size_t M = need_infer.size();
    if (M == 0) return ans;

    constexpr size_t img_bytes = 32 * 32 * sizeof(uint8_t);
    uint8_t* input_ptr = infer_request.get_input_tensor().data<uint8_t>();

    size_t processed = 0;
    while (processed < M) {
        size_t batch = std::min(MAX_BATCH, M - processed);

        for (size_t j = 0; j < MAX_BATCH; ++j) {
            uint8_t* dst = input_ptr + j * img_bytes;
            if (j < batch) {
                const cv::Mat& img = armors_pattern[need_infer[processed + j]];
                if (img.empty() || img.cols != 32 || img.rows != 32 || img.channels() != 1) {
                    std::memset(dst, 0, img_bytes);
                } else if (!img.isContinuous()) {
                    cv::Mat cont = img.clone();
                    std::memcpy(dst, cont.data, img_bytes);
                } else {
                    std::memcpy(dst, img.data, img_bytes);
                }
            } else {
                std::memset(dst, 0, img_bytes);
            }
        }

        infer_request.infer();

        const float* output = infer_request.get_output_tensor().data<float>();
        for (size_t i = 0; i < batch; ++i) {
            const float* logits = output + i * 9;

            float max_val = logits[0];
            int max_id = 0;
            for (int c = 1; c < 9; ++c) {
                if (logits[c] > max_val) { max_val = logits[c]; max_id = c; }
            }

            float sum = 0.0f;
            for (int c = 0; c < 9; ++c) sum += std::exp(logits[c] - max_val);
            ans[need_infer[processed + i]] = Ans(max_id, 1.0f / sum);
        }
        processed += batch;
    }

    return ans;
}

std::vector<ArmorPosi> ResNetNumClassifier::operator()(
    std::vector<std::array<ArmorPosi, 2>>& armors,
    const std::vector<cv::Mat>& armors_pattern)
{
    std::vector<ArmorPosi> result;
    if (armors.empty() || armors.size() != armors_pattern.size()) return result;

    std::vector<Ans> ans = Classify(armors,armors_pattern);
    result.reserve(ans.size());

    for (size_t i = 0; i < ans.size(); i++) {
        // 置信度不足，跳过
        if (ans[i].confidence <= confidence_threshold) continue;

        // 将模型输出 ID 映射到装甲板类型，同时确定大/小装甲板索引
        // idx=0: 小装甲板解算结果，idx=1: 大装甲板解算结果
        ArmorPosi::Type type = ArmorPosi::Type::Unknow;
        int idx = 0;
        //std::cout << "id: " << ans[i].id << ", confidence: " << ans[i].confidence << "\n";
        switch (ans[i].id) {
            case 0: type = ArmorPosi::Type::hero;    idx = 1; break; // 英雄，大装甲板
            case 1: type = ArmorPosi::Type::two;     idx = 0; break; // 工程
            case 2: type = ArmorPosi::Type::three;   idx = 0; break; // 3号步兵
            case 3: type = ArmorPosi::Type::four;    idx = 0; break; // 4号步兵
            case 5: type = ArmorPosi::Type::guard;   idx = 0; break; // 哨兵
            case 6: type = ArmorPosi::Type::outpost; idx = 0; break; // 前哨站
            case 7: type = ArmorPosi::Type::base;    idx = 1; break; // 基地，大装甲板
            default: continue; // case 4 (5号步兵，已弃用), case 8 (非装甲板)
        }

        //  标记该装甲板的位姿解算是否通过合法性检验
        if (!armors[i][idx].IsInRange) continue;
        result.emplace_back(armors[i][idx]);
        result.back().type = type;
        result.back().confidence = ans[i].confidence;
    }

    return result;
}
