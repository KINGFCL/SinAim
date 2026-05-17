#include "Config.hpp"
#include "Function.hpp"
#include "ResNetNumClassifier.hpp"
#include "libxr.hpp"
#include "linux_uart.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// #define MainDebug
#ifdef MainDebug
#include "communicate/RerunVisualizer.hpp"
RerunVisualizer viz("RoboMaster_AutoAim");
double R_sum = 0.0;
int R_count = 0;
#endif

namespace
{

constexpr const char* kConfigPath = "../../config/config.yaml";
constexpr const char* kModelPath = "../../model/tiny_resnet.onnx";
constexpr unsigned int kSerialBaud = 460800;

struct Test
{
    explicit Test(int cycle):cycle(cycle){};
private:
    int num = 0;
    const int cycle = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
public:
    void count()
    {
        ++this->num;
        if(this->num == this->cycle)
        {
            this->show();
            this->clear();
        }
        
    }

    void clear()
    {
        num = 0;
        this->start = std::chrono::steady_clock::now();
    }

    void show() const
    {
        auto total = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - this->start);
        std::cout << num / (static_cast<double>(total.count()) * 1e-9) << "Hz\n";
    }
};

static FastQueue<FrameData> Frames(10);
static FastQueue<std::unique_ptr<RobotState>> RobotStates(10);

std::chrono::steady_clock::time_point next_point = std::chrono::steady_clock::now();

CVDetector detect(LoadCVDetectorConfig(kConfigPath));
ResNetNumClassifier resnet(kModelPath);

Solver Sov(LoadSolverConfig(kConfigPath));
Tracker track(LoadRobotConfig(kConfigPath));
Shooter shoot(LoadShooterConfig(kConfigPath));
MPC::Planner planner(LoadPlannerConfig(kConfigPath));
Test test(200);

const char* ArmorLabelName(int id)
{
    switch (id) {
        case 0: return "one";
        case 1: return "two";
        case 2: return "three";
        case 3: return "four";
        case 4: return "five";
        case 5: return "sentry";
        case 6: return "outpost";
        case 7: return "base";
        case 8: return "not_armor";
        default: return "unknown";
    }
}

std::vector<ResNetNumClassifier::Ans> ClassifyDetectedPatterns(const std::vector<cv::Mat>& patterns)
{
    std::vector<std::array<ArmorPosi, 2>> armors;
    armors.reserve(patterns.size());

    std::array<ArmorPosi, 2> dummy_armor;
    dummy_armor[0].IsInRange = true;
    dummy_armor[1].IsInRange = true;
    for (size_t i = 0; i < patterns.size(); ++i) {
        armors.push_back(dummy_armor);
    }

    return resnet.Classify(armors, patterns);
}

std::vector<ArmorPosi> DetectArmors(cv::Mat& image, const Eigen::Quaterniond& gripper_to_world)
{
    std::vector<cv::Mat> armors_pattern;
    auto opencv_armors = detect(image, armors_pattern, CVDetector::ROIType::ResNet);
    std::cerr<< "opencv_armors:" << opencv_armors.size();
    auto armors_2 = Sov(opencv_armors, gripper_to_world);
    return resnet(armors_2, armors_pattern);
}
}  // namespace

int main(int argc, char** argv)
{
    #ifdef MainDebug
    #ifdef EKFKalmanDebug
    g_ekf_debug_cb = [](const Eigen::Matrix<double,14,1>& s,
                        const Eigen::Matrix<double,4,1>& v, double dt) {
        viz.EKFKalmanUpdate(s, v, dt);
    };
    #endif
    #endif
    const char* image_path = argc > 1 ? argv[1] : "/home/king/Desktop/debug32.png";

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << '\n';
        return 1;
    }

    std::vector<cv::Mat> armors_pattern;
    std::vector<CVArmor> cv_armors = detect(image, armors_pattern, CVDetector::ROIType::ResNet);

    cv::Mat debug_image = image.clone();
    detect.ArmorShow(debug_image, cv_armors);
    cv::imwrite("/home/king/Desktop/SinAim/rm-main/cvdetector_debug.png", debug_image);

    std::cout << "image: " << image_path << '\n';
    std::cout << "detected armors: " << cv_armors.size() << '\n';
    std::cout << "generated resnet patterns: " << armors_pattern.size() << '\n';
    std::cout << "debug image: /home/king/Desktop/SinAim/rm-main/cvdetector_debug.png\n";

    if (armors_pattern.empty()) {
        std::cerr << "No armor ROI detected by CVDetector\n";
        return 1;
    }

    for (size_t i = 0; i < armors_pattern.size(); ++i) {
        std::string roi_path = "/home/king/Desktop/SinAim/rm-main/resnet_roi_" + std::to_string(i) + ".png";
        cv::imwrite(roi_path, armors_pattern[i]);
    }

    std::vector<ResNetNumClassifier::Ans> results = ClassifyDetectedPatterns(armors_pattern);
    if (results.empty()) {
        std::cerr << "No classification result\n";
        return 1;
    }

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& ans = results[i];
        std::cout << "roi[" << i << "] "
                  << "id: " << ans.id
                  << ", label: " << ArmorLabelName(ans.id)
                  << ", confidence: " << ans.confidence
                  << ", image: /home/king/Desktop/SinAim/rm-main/resnet_roi_" << i << ".png\n";
    }

    return 0;
}
