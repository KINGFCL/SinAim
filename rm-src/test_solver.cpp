// 单元测试：验证 Solver 解算链路
// 核心问题：yaw 的定义和 +π 是否正确

#include <iostream>
#include <cmath>
#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "aim/include/Solver.hpp"
#include "identify/include/Armor.hpp"

// -------------------------------------------------------
// 工具：把 3D 点投影到图像
// -------------------------------------------------------
static std::vector<cv::Point2f> projectPoints3f(
    const std::vector<cv::Point3f>& pts3d,
    const cv::Mat& K, const cv::Mat& dist,
    const cv::Mat& rvec, const cv::Mat& tvec)
{
    std::vector<cv::Point2f> pts2d;
    cv::projectPoints(pts3d, rvec, tvec, K, dist, pts2d);
    return pts2d;
}

// -------------------------------------------------------
// 构造 CVArmor（直接替换 Lightcorners）
// -------------------------------------------------------
static CVArmor makeCVArmor(const std::vector<cv::Point2f>& corners)
{
    Light dummy(cv::RotatedRect(cv::Point2f(320,240), cv::Size2f(2,20), 90));
    CVArmor armor(dummy, dummy);
    armor.Lightcorners = corners;
    return armor;
}

// -------------------------------------------------------
// 测试 1：验证 yaw 的方向
//
// 设置：
//   - 相机坐标系 Z 轴 = 世界坐标系 X 轴（相机朝右）
//   - 装甲板在相机前方 300cm（= 世界 X 轴正方向 300cm）
//   - 装甲板无旋转（法向量指向相机，即 -X 方向）
//
// 期望：
//   - 装甲板世界坐标 ≈ (300, 0, 0)（Z 高度 ≈ 0，在范围内）
//   - 装甲板位置角 = atan2(0, 300) = 0 deg
//   - yaw 应该 ≈ 0 deg（和位置角一致，EKF 才能正确工作）
// -------------------------------------------------------
void test_yaw_vs_position_angle()
{
    std::cout << "\n=== test_yaw_vs_position_angle ===\n";

    // 用 697e64c 版本的旧相机参数
    Solver::SolverConfig cfg;
    cfg.camera_matrix = {
        2812.551261951036, 0, 638.9266938290903,
        0, 2784.250529904723, 347.0750616634094,
        0, 0, 1
    };
    cfg.distortion_coeffs = {
        -0.2102688173489966, 3.12388655173893,
        0.008022463142918946, -0.007259818830423669, -16.88025730703067
    };
    // R_Cam_to_gripper: 相机 Z 轴 ≈ 云台 X 轴（相机朝右安装）
    // 用近似值：相机 Z = 世界 X，相机 X = 世界 -Y，相机 Y = 世界 -Z
    // R_Cam_to_gripper 按行展开：
    //   [0, -1, 0]   (相机X -> 云台Y的负方向)
    //   [0,  0, -1]  (相机Y -> 云台Z的负方向)
    //   [1,  0,  0]  (相机Z -> 云台X)
    cfg.R_Cam_to_gripper = {
         0, -1,  0,
         0,  0, -1,
         1,  0,  0
    };
    cfg.T_Cam_to_gripper = {0, 0, 0};
    cfg.reproj_threshold = 1.0;

    Solver solver(cfg);

    cv::Mat K = (cv::Mat_<double>(3,3) <<
        cfg.camera_matrix[0], 0, cfg.camera_matrix[2],
        0, cfg.camera_matrix[4], cfg.camera_matrix[5],
        0, 0, 1);
    cv::Mat dist = (cv::Mat_<double>(5,1) <<
        cfg.distortion_coeffs[0], cfg.distortion_coeffs[1],
        cfg.distortion_coeffs[2], cfg.distortion_coeffs[3],
        cfg.distortion_coeffs[4]);

    // 小装甲板 3D 点（以中心为原点）
    float hs = 5.5f / 2.0f, ws = 13.5f / 2.0f;
    std::vector<cv::Point3f> pts3d = {
        {-ws, -hs, 0}, {ws, -hs, 0}, {ws, hs, 0}, {-ws, hs, 0}
    };

    // 装甲板在相机前方 300cm，无旋转
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = (cv::Mat_<double>(3,1) << 0.0, 0.0, 300.0);

    auto corners = projectPoints3f(pts3d, K, dist, rvec, tvec);
    std::cout << "投影角点: ";
    for (auto& p : corners) std::cout << "(" << p.x << "," << p.y << ") ";
    std::cout << "\n";

    CVArmor armor = makeCVArmor(corners);

    // 云台姿态为单位四元数（云台坐标系 = 世界坐标系）
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();

    auto results = solver({armor}, q);

    if (results.empty()) {
        std::cout << "FAIL: 解算结果为空\n";
        return;
    }

    auto& small = results[0][0];
    std::cout << "IsInRange: " << small.IsInRange << "\n";
    std::cout << "解 0 中心(世界系): " << small.center.col(0).transpose() << "\n";
    std::cout << "解 1 中心(世界系): " << small.center.col(1).transpose() << "\n";
    std::cout << "reproj[0]: " << small.reproj[0] << "  reproj[1]: " << small.reproj[1] << "\n";
    std::cout << "yaw[0]: " << small.yaw[0] * 180.0 / M_PI << " deg\n";
    std::cout << "yaw[1]: " << small.yaw[1] * 180.0 / M_PI << " deg\n";
    std::cout << "yaw_abs[0]: " << small.yaw_abs[0] * 180.0 / M_PI << " deg\n";
    std::cout << "theta[0]: " << small.theta[0] * 180.0 / M_PI << " deg\n";

    if (!small.IsInRange) {
        std::cout << "FAIL: 不在范围内\n";
        return;
    }

    // 期望：装甲板在世界 X 轴正方向，位置角 = 0 deg
    // 正确解的中心应该接近 (300, 0, 0)
    Eigen::Vector3d expected(300.0, 0.0, 0.0);
    double err0 = (small.center.col(0) - expected).norm();
    double err1 = (small.center.col(1) - expected).norm();
    std::cout << "解 0 位置误差: " << err0 << " cm\n";
    std::cout << "解 1 位置误差: " << err1 << " cm\n";

    int correct_side = (err0 < err1) ? 0 : 1;
    double correct_yaw = small.yaw[correct_side];
    double position_angle = std::atan2(expected(1), expected(0));  // = 0

    double yaw_err = std::abs(std::remainder(correct_yaw - position_angle, 2*M_PI));
    std::cout << "正确解 yaw: " << correct_yaw * 180.0 / M_PI << " deg\n";
    std::cout << "期望位置角: " << position_angle * 180.0 / M_PI << " deg\n";
    std::cout << "yaw 与位置角差: " << yaw_err * 180.0 / M_PI << " deg\n";

    if (yaw_err < 0.2) {
        std::cout << "PASS: yaw 与位置角一致\n";
    } else if (std::abs(yaw_err - M_PI) < 0.2) {
        std::cout << "FAIL: yaw 与位置角差 180 deg（方向反了）\n";
    } else {
        std::cout << "FAIL: yaw 与位置角差 " << yaw_err * 180.0 / M_PI << " deg\n";
    }
}

// -------------------------------------------------------
// 测试 2：验证 ArmorPosi 构造函数的 yaw 处理
//
// 直接构造 ArmorPosi，验证 yaw 是否正确
// -------------------------------------------------------
void test_armor_posi_yaw()
{
    std::cout << "\n=== test_armor_posi_yaw ===\n";

    // 场景：装甲板在世界坐标系 (300, 0, 0)
    // 装甲板 X 轴（宽度方向）在世界坐标系下是 Y 轴方向
    // 所以 yaw_input = atan2(0, 1) = 0（X 轴方向）
    // 但实际上装甲板宽度方向是 Y 轴，yaw_input = atan2(1, 0) = 90 deg

    // 先测试最简单的情况：装甲板正对 X 轴，宽度方向是 Y 轴
    // yaw_input = atan2(toward_world.y, toward_world.x)
    // toward_world = R_cam2world * R_armor * (1,0,0)
    // 相机 Z = 世界 X，相机 X = 世界 -Y，相机 Y = 世界 -Z
    // R_cam2world = [[0,-1,0],[0,0,-1],[1,0,0]]^T = [[0,0,1],[-1,0,0],[0,-1,0]]
    // R_armor = I（无旋转）
    // toward_world = R_cam2world * (1,0,0) = (0,-1,0)（世界 -Y 方向）
    // yaw_input = atan2(-1, 0) = -90 deg

    // 装甲板位置角 = atan2(0, 300) = 0 deg
    // 如果 yaw_input = -90 deg，加 π 后 = 90 deg，和位置角 0 deg 差 90 deg -> WRONG
    // 如果 yaw_input = -90 deg，不加 π，= -90 deg，和位置角 0 deg 差 90 deg -> WRONG

    // 这说明用 X 轴（宽度方向）计算 yaw 本身就是错的！
    // 应该用 Z 轴（法向量）：
    // toward_world = R_cam2world * (0,0,1) = (1,0,0)（世界 X 方向）
    // yaw_input = atan2(0, 1) = 0 deg
    // 装甲板位置角 = 0 deg -> 一致！不需要加 π

    std::cout << "分析：\n";
    // R_cam2world（相机Z=世界X，相机X=世界-Y，相机Y=世界-Z）
    Eigen::Matrix3d R_cam2world;
    R_cam2world << 0, 0, 1,
                  -1, 0, 0,
                   0,-1, 0;

    Eigen::Vector3d x_axis_cam(1, 0, 0);
    Eigen::Vector3d z_axis_cam(0, 0, 1);

    Eigen::Vector3d toward_x = R_cam2world * x_axis_cam;
    Eigen::Vector3d toward_z = R_cam2world * z_axis_cam;

    double yaw_x = std::atan2(toward_x(1), toward_x(0));
    double yaw_z = std::atan2(toward_z(1), toward_z(0));

    std::cout << "  用 X 轴计算 yaw: " << yaw_x * 180.0 / M_PI << " deg\n";
    std::cout << "  用 Z 轴计算 yaw: " << yaw_z * 180.0 / M_PI << " deg\n";
    std::cout << "  装甲板位置角: 0 deg\n";

    double pos_angle = 0.0;
    double err_x = std::abs(std::remainder(yaw_x - pos_angle, 2*M_PI));
    double err_z = std::abs(std::remainder(yaw_z - pos_angle, 2*M_PI));
    double err_x_pi = std::abs(std::remainder(yaw_x + M_PI - pos_angle, 2*M_PI));
    double err_z_pi = std::abs(std::remainder(yaw_z + M_PI - pos_angle, 2*M_PI));

    std::cout << "  X轴yaw 与位置角差: " << err_x * 180.0 / M_PI << " deg\n";
    std::cout << "  X轴yaw+π 与位置角差: " << err_x_pi * 180.0 / M_PI << " deg\n";
    std::cout << "  Z轴yaw 与位置角差: " << err_z * 180.0 / M_PI << " deg\n";
    std::cout << "  Z轴yaw+π 与位置角差: " << err_z_pi * 180.0 / M_PI << " deg\n";

    if (err_x < 0.1) std::cout << "  -> X轴yaw 直接等于位置角 (PASS)\n";
    else if (err_x_pi < 0.1) std::cout << "  -> X轴yaw+π 等于位置角 (需要+π)\n";
    else std::cout << "  -> X轴yaw 与位置角不一致 (FAIL)\n";

    if (err_z < 0.1) std::cout << "  -> Z轴yaw 直接等于位置角 (PASS)\n";
    else if (err_z_pi < 0.1) std::cout << "  -> Z轴yaw+π 等于位置角 (需要+π)\n";
    else std::cout << "  -> Z轴yaw 与位置角不一致 (FAIL)\n";
}

// -------------------------------------------------------
// 测试 3：用真实相机参数验证
// -------------------------------------------------------
void test_real_params()
{
    std::cout << "\n=== test_real_params (697e64c 参数) ===\n";

    // 697e64c 的 R_Cam_to_gripper（按行展开）
    // 第一行 ≈ (0.0016, 0.0474, 0.9989) -> 相机X轴 ≈ 世界Z轴
    // 第二行 ≈ (-0.9983, 0.0575, -0.0012) -> 相机Y轴 ≈ -世界X轴
    // 第三行 ≈ (-0.0575, -0.9972, 0.0474) -> 相机Z轴 ≈ -世界Y轴
    // 所以相机朝向 ≈ -Y 方向（向左）

    Eigen::Matrix3d R_cam_to_grip;
    R_cam_to_grip << 0.001575954905394993, 0.04738879556158539, 0.9988752767094389,
                    -0.9983451829858779, 0.05749406315477609, -0.001152523686676228,
                    -0.05748401495224464, -0.9972205045811972, 0.04740098382727122;

    // 云台姿态为单位四元数
    Eigen::Matrix3d R_gripper_to_world = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_cam2world = R_gripper_to_world * R_cam_to_grip;

    std::cout << "R_cam2world:\n" << R_cam2world << "\n";

    // 装甲板在相机前方 300cm，无旋转
    // 相机 Z 轴在世界坐标系下的方向
    Eigen::Vector3d cam_z_in_world = R_cam2world.col(2);
    std::cout << "相机Z轴(前方)在世界坐标系: " << cam_z_in_world.transpose() << "\n";
    std::cout << "装甲板世界坐标 ≈ 300 * " << cam_z_in_world.transpose() << "\n";

    Eigen::Vector3d armor_world = 300.0 * cam_z_in_world;
    double position_angle = std::atan2(armor_world(1), armor_world(0));
    std::cout << "装甲板位置角: " << position_angle * 180.0 / M_PI << " deg\n";

    // 用 X 轴计算 yaw（697e64c 的方式）
    Eigen::Vector3d toward_x = R_cam2world * Eigen::Vector3d(1, 0, 0);
    double yaw_x = std::atan2(toward_x(1), toward_x(0));
    std::cout << "X轴yaw: " << yaw_x * 180.0 / M_PI << " deg\n";
    std::cout << "X轴yaw+π: " << std::remainder(yaw_x + M_PI, 2*M_PI) * 180.0 / M_PI << " deg\n";

    // 用 Z 轴计算 yaw（当前代码的方式）
    Eigen::Vector3d toward_z = R_cam2world * Eigen::Vector3d(0, 0, 1);
    double yaw_z = std::atan2(toward_z(1), toward_z(0));
    std::cout << "Z轴yaw: " << yaw_z * 180.0 / M_PI << " deg\n";
    std::cout << "Z轴yaw+π: " << std::remainder(yaw_z + M_PI, 2*M_PI) * 180.0 / M_PI << " deg\n";

    double err_x = std::abs(std::remainder(yaw_x - position_angle, 2*M_PI));
    double err_x_pi = std::abs(std::remainder(yaw_x + M_PI - position_angle, 2*M_PI));
    double err_z = std::abs(std::remainder(yaw_z - position_angle, 2*M_PI));
    double err_z_pi = std::abs(std::remainder(yaw_z + M_PI - position_angle, 2*M_PI));

    std::cout << "\n与位置角的差:\n";
    std::cout << "  X轴yaw: " << err_x * 180.0 / M_PI << " deg\n";
    std::cout << "  X轴yaw+π: " << err_x_pi * 180.0 / M_PI << " deg\n";
    std::cout << "  Z轴yaw: " << err_z * 180.0 / M_PI << " deg\n";
    std::cout << "  Z轴yaw+π: " << err_z_pi * 180.0 / M_PI << " deg\n";

    std::cout << "\n结论:\n";
    auto check = [](double err, const char* name) {
        if (err < 5.0) std::cout << "  PASS: " << name << " 与位置角一致\n";
        else std::cout << "  FAIL: " << name << " 与位置角差 " << err << " deg\n";
    };
    check(err_x * 180.0 / M_PI, "X轴yaw (不加π)");
    check(err_x_pi * 180.0 / M_PI, "X轴yaw+π");
    check(err_z * 180.0 / M_PI, "Z轴yaw (不加π)");
    check(err_z_pi * 180.0 / M_PI, "Z轴yaw+π");
}

int main()
{
    std::cout << "========== Solver yaw 方向单元测试 ==========\n";
    test_yaw_vs_position_angle();
    test_armor_posi_yaw();
    test_real_params();
    std::cout << "\n========== 测试完成 ==========\n";
    return 0;
}
