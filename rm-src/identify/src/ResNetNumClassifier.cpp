#include "ResNetNumClassifier.hpp"
#include "Armor.hpp"
#include <iostream>
#include <cmath>

static const size_t MAX_BATCH_SIZE = 4;
static const size_t GPU_request_num = 2;
static const size_t CPU_request_num = 12;
static const size_t GPU_ENABLE_THRESHOLD = 8; // 目标数量大于8时优先塞给GPU

ResNetNumClassifier::ResNetNumClassifier(std::string model_path, float confidence_threshold)
    : confidence_threshold(confidence_threshold)
{
    std::shared_ptr<ov::Model> model_base = core.read_model(model_path);

    // ==========================================
    // 预处理融合：让模型内部直接吃 uint8 数据并自动除以 255
    // ==========================================
    ov::preprocess::PrePostProcessor ppp(model_base);
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NCHW"); // 单通道 32x32，即 [Batch, 1, 32, 32]
    ppp.input().preprocess()
        .convert_element_type(ov::element::f32)
        .scale(255.0f); // 相当于 1.0 / 255.0
    
    std::shared_ptr<ov::Model> model = ppp.build();

    // ==========================================
    // CPU 模型配置 (Batch = 1)
    // ==========================================
    auto model_CPU = model->clone();
    std::map<std::string, ov::PartialShape> shapes_cpu;
    shapes_cpu[model_CPU->input().get_any_name()] = ov::PartialShape{1, 1, 32, 32}; 
    model_CPU->reshape(shapes_cpu);

    // ==========================================
    // GPU 模型配置 (Batch = MAX_BATCH_SIZE)
    // ==========================================
    auto model_GPU = model->clone();
    std::map<std::string, ov::PartialShape> shapes_gpu;
    shapes_gpu[model_GPU->input().get_any_name()] = ov::PartialShape{MAX_BATCH_SIZE, 1, 32, 32}; 
    model_GPU->reshape(shapes_gpu);

    // 编译 GPU 模型
    try {
        ov::AnyMap gpu_props;
        gpu_props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        gpu_props[ov::cache_dir.name()] = "./gpu_cache";
        compiled_model_GPU = core.compile_model(model_GPU, "GPU", gpu_props);
        has_gpu = true;
        for(int i = 0 ; i < GPU_request_num; i++)
            infer_request_GPU.emplace_back(compiled_model_GPU.create_infer_request());
        std::cout << "[分类器] 成功加载到 GPU工位: " << GPU_request_num << "个" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[分类器] GPU加载失败: " << e.what() << std::endl;
    }

    // 编译 CPU 模型
    try {
        ov::AnyMap cpu_props;
        cpu_props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        compiled_model_CPU = core.compile_model(model_CPU, "CPU", cpu_props);
        has_cpu = true;
        for(int i = 0 ; i < CPU_request_num; i++)
            infer_request_CPU.emplace_back(compiled_model_CPU.create_infer_request());
        std::cout << "[分类器] 成功加载到 CPU工位: " << CPU_request_num << "个" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[分类器] 致命错误: CPU加载失败" << std::endl;
        exit(-1);
    }
}

std::vector<ResNetNumClassifier::Ans> ResNetNumClassifier::Classify(const std::vector<cv::Mat>& armors_pattern)
{
    size_t N = armors_pattern.size();
    std::vector<Ans> ans(N); // 预分配大小
    if (N == 0) return ans;

    // 任务追踪结构体
    struct Task {
        ov::InferRequest* req;
        size_t start_idx;
        size_t count;
    };
    std::vector<Task> tasks;

    size_t img_idx = 0;
    size_t gpu_idx = 0;
    size_t cpu_idx = 0;
    size_t num = N; 
    const size_t img_bytes = 32 * 32 * sizeof(uint8_t); // 每张图 1024 字节

    // 【修复3】：在循环外判定是否优先使用 GPU，防止后续 num 递减导致 GPU 饿死
    bool prioritize_gpu = (N >= GPU_ENABLE_THRESHOLD);

    // ==========================================
    // 阶段 1：多路分发 (Fork) 
    // ==========================================
    while (num > 0) {
        ov::InferRequest* req = nullptr;
        size_t count = 0;

        // 调度器：使用全局 flag 判定 GPU 优先级
        if (prioritize_gpu && gpu_idx < infer_request_GPU.size()) {
            req = &infer_request_GPU[gpu_idx++];
            count = std::min(MAX_BATCH_SIZE, num);
        } else if (cpu_idx < infer_request_CPU.size()) {
            req = &infer_request_CPU[cpu_idx++];
            count = 1;
        } else if (gpu_idx < infer_request_GPU.size()) { // Fallback：如果 CPU 满了但 GPU 还有空位
            req = &infer_request_GPU[gpu_idx++];
            count = std::min(MAX_BATCH_SIZE, num);
        } else {
            std::cerr << "[警告] 目标过多，超出并发池容量，截断处理！" << "\n";
            break; 
        }

        // 【修复2】：直接获取当前 Tensor 的期望 Batch 容量，精准定长循环防止越界
        uint8_t* input_ptr = req->get_input_tensor().data<uint8_t>();
        size_t tensor_batch_capacity = req->get_input_tensor().get_shape()[0];

        for (size_t j = 0; j < tensor_batch_capacity; ++j) {
            if (j < count) {
                const cv::Mat& img = armors_pattern[img_idx + j];
                
                // 健壮性校验
                if (img.empty() || img.cols != 32 || img.rows != 32 || img.channels() != 1) {
                    std::memset(input_ptr + j * img_bytes, 0, img_bytes);
                } 
                // 【修复1】：强制连续内存判定，规避 ROI 截取带来的脏数据和段错误
                else if (!img.isContinuous() || img.total() * img.elemSize() != img_bytes) {
                    cv::Mat continuous_img = img.clone();
                    std::memcpy(input_ptr + j * img_bytes, continuous_img.data, img_bytes);
                } 
                else {
                    std::memcpy(input_ptr + j * img_bytes, img.data, img_bytes);
                }
            } else {
                // 踏踏实实填充黑图 Padding，防止随机内存干扰网络
                std::memset(input_ptr + j * img_bytes, 0, img_bytes);
            }
        }

        // 发射！不阻塞主线程
        req->start_async();
        tasks.push_back({req, img_idx, count});
        
        num -= count;
        img_idx += count;
    }

    // ==========================================
    // 阶段 2：收集与极速后处理 (Join)
    // ==========================================
    for (const auto& task : tasks) {
        task.req->wait(); // 阻塞等待当前工位完成
        
        // 获取输出 [batch_size, 9] 的裸指针
        const float* output_data = task.req->get_output_tensor().data<float>();

        for (size_t i = 0; i < task.count; ++i) {
            const float* logits = output_data + i * 9;
            
            // 手动 SIMD 友好的 Softmax 解析
            float max_val = logits[0];
            int max_id = 0;
            for (int c = 1; c < 9; ++c) {
                if (logits[c] > max_val) {
                    max_val = logits[c];
                    max_id = c;
                }
            }

            float sum = 0.0f;
            for (int c = 0; c < 9; ++c) {
                sum += std::exp(logits[c] - max_val);
            }
            float confidence = 1.0f / sum; // exp(max_val - max_val) == 1，直接用倒数

            ans[task.start_idx + i] = Ans(max_id, confidence);
        }
    }

    return ans;
}

std::vector<ArmorPosi> ResNetNumClassifier::operator()(std::vector<std::array<ArmorPosi, 2>>& armors, const std::vector<cv::Mat>& armors_pattern, const std::vector< std::array<bool, 2> >& PosePassHax)
{
    std::vector<ArmorPosi> result;

    // 【修复4】：强校验，防止后续 `armors[i]` 出现内存越界
    if (armors.empty() || armors.size() != armors_pattern.size()) {
        if (!armors.empty()) {
            std::cerr << "[错误] 装甲板坐标数量 (" << armors.size() 
                      << ") 与图案数量 (" << armors_pattern.size() << ") 不匹配！" << std::endl;
        }
        return result;
    }

    std::vector<Ans> ans = Classify(armors_pattern);
    result.reserve(ans.size());

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
    * | 8             | not_armor           | 非装甲板 (背景/误检)        |
    */
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
                case 5: // 模型 5 = 哨兵 (sentry)
                    final_type = ArmorPosi::Type::guard;
                    target_idx = 0;
                    break;
                case 6: // 模型 6 = 前哨站 (outpost)
                    final_type = ArmorPosi::Type::outpost;
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
            if (final_type == ArmorPosi::Type::Unknow || !PosePassHax[i][target_idx]) continue;

            // 4. 压入结果
            result.emplace_back(armors[i][target_idx]);
            result.back().type = final_type;
            result.back().confidence = ans[i].confidence;
        }
    }
    
    return result;
}