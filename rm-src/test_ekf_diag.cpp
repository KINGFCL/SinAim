// 诊断测试：EKF 半径缩小问题
// 模拟一个已知参数的旋转机器人，逐帧输出 Solver 和 EKF 的中间值
//
// 编译方式（在 build 目录下）：
//   cmake .. && make test_ekf_diag
// 或直接：
//   g++ -std=c++17 -O0 -I.. $(pkg-config --cflags --libs opencv4 eigen3) \
//       aim/src/Solver.cpp aim/src/EKFKalman.cpp identify/src/Identify.cpp \
//       test_ekf_diag.cpp -o test_ekf_diag

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "Solver.hpp"
#include "EKFKalman.hpp"
#include "Armor.hpp"

static CVArmor makeCVArmor(const std::vector<cv::Point2f>& corners)
{
    Light dummy(cv::RotatedRect(cv::Point2f(320,240), cv::Size2f(2,20), 90));
    CVArmor armor(dummy, dummy);
    armor.Lightcorners = corners;
    return armor;
}

int main()
{
    std::cout << std::fixed << std::setprecision(3);

    // ── 相机参数（来自 config/solver.yaml）──────────────────────────────
    Solver::SolverConfig cfg;
    cfg.camera_matrix = {2328.685719898089, 0, 733.3564625092474,
                         0, 2328.670107789996, 540.6187286922773,
                         0, 0, 1};
    cfg.distortion_coeffs = {-0.09182103918709904, 0.4639907346830205,
                              0.002609878642637282, 0.0009819586010405485,
                              -0.4751278850310457};
    cfg.R_Cam_to_gripper = {-0.009549480539577278, -0.01953893000739315, 0.9997634908495061,
                             -0.9999090215267193, -0.009338627954961053, -0.009733380573965271,
                              0.009526599125766769, -0.9997654826218425, -0.01944797333944928};
    cfg.T_Cam_to_gripper = {13.6068364765315, -4.186176466382783, 0.8995665883635868};
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

    // R_cam2world（云台=世界，单位四元数）
    Eigen::Matrix3d R_c2g;
    R_c2g << cfg.R_Cam_to_gripper[0], cfg.R_Cam_to_gripper[1], cfg.R_Cam_to_gripper[2],
             cfg.R_Cam_to_gripper[3], cfg.R_Cam_to_gripper[4], cfg.R_Cam_to_gripper[5],
             cfg.R_Cam_to_gripper[6], cfg.R_Cam_to_gripper[7], cfg.R_Cam_to_gripper[8];
    Eigen::Vector3d T_c2g(cfg.T_Cam_to_gripper[0], cfg.T_Cam_to_gripper[1], cfg.T_Cam_to_gripper[2]);

    Eigen::Matrix3d R_world2cam = R_c2g.transpose();
    Eigen::Vector3d photocenter_world = T_c2g; // 相机光心在世界坐标系

    // ── 仿真参数 ──────────────────────────────────────────────────────────
    // 机器人中心在世界坐标系 (200, 0, 0) cm，正对相机
    const double robot_cx = 200.0, robot_cy = 0.0, robot_cz = 0.0;
    const double armor_r  = 24.0;   // 装甲板半径 cm
    const double omega    = 1.0;    // 角速度 rad/s
    const double dt       = 1.0/30.0;

    // 小装甲板尺寸（与 Solver 内部一致）
    const float hs = 5.5f/2.0f, ws = 13.5f/2.0f;
    // 装甲板本地坐标系下的四个角点（与 Solver objectSmallArmorP 完全一致）
    std::vector<cv::Point3f> pts3d = {
        {-ws, -hs, 0}, {ws, -hs, 0}, {ws, hs, 0}, {-ws, hs, 0}
    };

    // ── EKF ──────────────────────────────────────────────────────────────
    EKFKalman ekf;
    ekf.Init();
    Eigen::Matrix<double, 14, 1> State = Eigen::Matrix<double, 14, 1>::Zero();
    bool ekf_initialized = false;

    std::cout << "========== EKF 半径诊断测试 ==========\n\n";

    for (int frame = 0; frame < 40; frame++)
    {
        double theta = omega * frame * dt; // 真实装甲板 0 的位置角

        // ── 真实装甲板位置（世界坐标系）────────────────────────────────
        double ax = robot_cx + armor_r * std::cos(theta);
        double ay = robot_cy + armor_r * std::sin(theta);
        double az = robot_cz;

        // ── 装甲板在世界坐标系下的旋转矩阵 ──────────────────────────────
        // Z 轴（法向量，朝外）= (cos θ, sin θ, 0)
        // Y 轴（向上）        = (0, 0, 1)
        // X 轴（向右，从正面看）= Y × Z
        //   Y × Z = (0,0,1) × (cosθ,sinθ,0) = (-sinθ, cosθ, 0)
        //   但相机 X（右）≈ -World Y，所以从相机看装甲板"右"方向 = -World Y 方向
        //   对于 θ=0，-World Y = (0,-1,0)，而 Y×Z = (0,1,0)，方向相反
        //   因此 X = -(Y × Z) = (sinθ, -cosθ, 0) 才能让正 X 对应相机右侧
        //   但这会让坐标系变成左手系！正确做法：用 Z × Y
        //   Z × Y = (cosθ,sinθ,0) × (0,0,1) = (sinθ*1-0*0, 0*0-cosθ*1, cosθ*0-sinθ*0)
        //         = (sinθ, -cosθ, 0)
        //   验证右手系：X × Y = Z?  (sinθ,-cosθ,0)×(0,0,1) = (-cosθ*1-0, 0-sinθ*1, 0)
        //                          = (-cosθ, -sinθ, 0) ≠ Z  -> 不是右手系
        //   正确右手系：X = Y × Z 的叉积
        //   Y × Z = (0,0,1)×(cosθ,sinθ,0) = (0*0-1*sinθ, 1*cosθ-0*0, 0*sinθ-0*cosθ)
        //         = (-sinθ, cosθ, 0)
        //   验证：X × Y = (-sinθ,cosθ,0)×(0,0,1) = (cosθ*1-0*0, 0*0-(-sinθ)*1, (-sinθ)*0-cosθ*0)
        //              = (cosθ, sinθ, 0) = Z ✓  右手系成立
        //   对于 θ=0：X=(-sinθ,cosθ,0)=(0,1,0)=World Y
        //   相机 X（右）≈ -World Y，所以 World Y 对应相机左侧
        //   这意味着装甲板 +X 方向（宽度正方向）在图像左侧
        //   objectPoints 中 corner[0]=(-ws,-hs,0) 是左上，corner[1]=(ws,-hs,0) 是右上
        //   "右上"在图像左侧，"左上"在图像右侧 -> 左右互换
        //   要让 corner[1] 在图像右侧，需要 X = -World Y 方向 = (sinθ,-cosθ,0)
        //   但这破坏右手系。解决方案：翻转 Y 轴，让 Y = (0,0,-1)（向下）
        //   则 X = Y × Z = (0,0,-1)×(cosθ,sinθ,0) = (0*0-(-1)*sinθ, (-1)*cosθ-0*0, 0*sinθ-0*cosθ)
        //               = (sinθ, -cosθ, 0)
        //   验证：X × Y = (sinθ,-cosθ,0)×(0,0,-1) = (-cosθ*(-1)-0*0, 0*sinθ-sinθ*(-1), sinθ*0-(-cosθ)*sinθ)
        //              = (cosθ, sinθ, 0) = Z ✓
        //   对于 θ=0：X=(0,-1,0)=-World Y ≈ 相机 X（右）✓  Y=(0,0,-1)=相机 Y（下）✓
        Eigen::Matrix3d R_armor_world;
        R_armor_world.col(0) = Eigen::Vector3d( std::sin(theta), -std::cos(theta), 0); // X（向右）
        R_armor_world.col(1) = Eigen::Vector3d(0, 0, -1);                              // Y（向下）
        R_armor_world.col(2) = Eigen::Vector3d( std::cos(theta),  std::sin(theta), 0); // Z（法向量）

        // ── 转换到相机坐标系 ─────────────────────────────────────────────
        Eigen::Vector3d armor_world(ax, ay, az);
        Eigen::Vector3d armor_cam = R_world2cam * (armor_world - photocenter_world);
        Eigen::Matrix3d R_armor_cam = R_world2cam * R_armor_world;

        // 检查装甲板是否在相机前方
        if (armor_cam(2) <= 0) {
            std::cout << "[frame " << frame << "] 装甲板在相机后方，跳过\n";
            continue;
        }

        // ── 生成 rvec/tvec 并投影角点 ────────────────────────────────────
        cv::Mat R_cv(3, 3, CV_64F);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                R_cv.at<double>(r,c) = R_armor_cam(r,c);
        cv::Mat rvec, tvec = (cv::Mat_<double>(3,1) << armor_cam(0), armor_cam(1), armor_cam(2));
        cv::Rodrigues(R_cv, rvec);

        std::vector<cv::Point2f> corners;
        cv::projectPoints(pts3d, rvec, tvec, K, dist, corners);

        // 检查角点是否在图像内
        bool in_image = true;
        for (auto& p : corners)
            if (p.x < 0 || p.x > 1280 || p.y < 0 || p.y > 1024) { in_image = false; break; }
        if (!in_image) {
            std::cout << "[frame " << frame << "] 角点超出图像范围，跳过\n";
            continue;
        }

        // ── 调用 Solver ──────────────────────────────────────────────────
        CVArmor cv_armor = makeCVArmor(corners);
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        auto results = solver({cv_armor}, q);

        std::cout << "── frame " << std::setw(2) << frame
                  << "  GT: theta=" << std::setw(7) << theta*180/M_PI << "deg"
                  << "  armor_pos=(" << ax << "," << ay << "," << az << ")\n";

        if (results.empty()) {
            std::cout << "   Solver 返回空结果\n\n";
            continue;
        }

        auto& ap = results[0][0]; // 小装甲板
        if (!ap.IsInRange) {
            std::cout << "   Solver: IsInRange=false  reproj[0]=" << ap.reproj[0]
                      << "  reproj[1]=" << ap.reproj[1] << "\n\n";
            continue;
        }

        // 真实 yaw（Z 轴方向 = 位置角）
        double gt_yaw = theta;

        std::cout << "   Solver sol0: center=(" << ap.center(0,0) << "," << ap.center(1,0) << "," << ap.center(2,0) << ")"
                  << "  yaw=" << ap.yaw[0]*180/M_PI << "deg"
                  << "  reproj=" << ap.reproj[0] << "\n";
        std::cout << "   Solver sol1: center=(" << ap.center(0,1) << "," << ap.center(1,1) << "," << ap.center(2,1) << ")"
                  << "  yaw=" << ap.yaw[1]*180/M_PI << "deg"
                  << "  reproj=" << ap.reproj[1] << "\n";
        std::cout << "   GT yaw=" << gt_yaw*180/M_PI << "deg"
                  << "  theta[0]=" << ap.theta[0]*180/M_PI << "deg"
                  << "  yaw_abs[0]=" << ap.yaw_abs[0]*180/M_PI << "deg\n";

        double yaw_err0 = std::abs(std::remainder(ap.yaw[0] - gt_yaw, 2*M_PI)) * 180/M_PI;
        double yaw_err1 = std::abs(std::remainder(ap.yaw[1] - gt_yaw, 2*M_PI)) * 180/M_PI;
        std::cout << "   yaw误差: sol0=" << yaw_err0 << "deg  sol1=" << yaw_err1 << "deg\n";

        // ── EKF 更新 ─────────────────────────────────────────────────────
        Eigen::Matrix<double, 4, 1> View;
        View(0) = ap.center(0, 0);
        View(1) = ap.center(1, 0);
        View(2) = ap.center(2, 0);
        View(3) = ap.yaw[0];

        if (!ekf_initialized) {
            State.setZero();
            State(0) = ap.center(0,0) - armor_r * std::cos(ap.yaw[0]);
            State(1) = ap.center(1,0) - armor_r * std::sin(ap.yaw[0]);
            State(2) = ap.center(2,0);
            State(6) = ap.yaw[0];
            State(7) = 0.0;
            State(8) = armor_r;
            State(9) = 0.0;
            State(10) = 0.0;
            State(11) = M_PI/2;
            State(12) = M_PI;
            State(13) = -M_PI/2;
            ekf_initialized = true;
            std::cout << "   EKF INIT: center=(" << State(0) << "," << State(1) << "," << State(2) << ")"
                      << "  r=" << State(8) << "  theta0=" << State(6)*180/M_PI << "deg\n";
        } else {
            // 预测的 armor_angle（armor id=0）
            double pred_armor_angle = State(6); // theta_0
            double pred_armor_x = State(0) + State(8) * std::cos(pred_armor_angle);
            double pred_armor_y = State(1) + State(8) * std::sin(pred_armor_angle);
            std::cout << "   EKF预测: armor_angle=" << pred_armor_angle*180/M_PI << "deg"
                      << "  pred_pos=(" << pred_armor_x << "," << pred_armor_y << ")\n";
            std::cout << "   EKF观测: yaw=" << View(3)*180/M_PI << "deg"
                      << "  pos=(" << View(0) << "," << View(1) << ")\n";
            std::cout << "   yaw残差=" << std::remainder(View(3)-pred_armor_angle, 2*M_PI)*180/M_PI << "deg\n";

            State = ekf(State, View, ap.SCS.block<3,1>(0,0), ap.yaw_abs[0], 0, dt);

            std::cout << "   EKF更新后: center=(" << State(0) << "," << State(1) << "," << State(2) << ")"
                      << "  r=" << State(8) << "  theta0=" << State(6)*180/M_PI << "deg"
                      << "  w=" << State(7) << "rad/s\n";
            std::cout << "   GT center=(" << robot_cx << "," << robot_cy << "," << robot_cz << ")"
                      << "  r=" << armor_r << "  theta0=" << theta*180/M_PI << "deg\n";
        }
        std::cout << "\n";
    }

    std::cout << "========== 测试完成 ==========\n";
    return 0;
}
