#include "Solver.hpp"
// #include <array>
#include <Eigen/src/Core/Matrix.h>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <numeric>

// #define SolverDebug
#ifdef SolverDebug
#include "RerunVisualizer.hpp"
extern RerunVisualizer viz;
#endif

namespace SASize {
    constexpr double LIGHTBAR_LENGTH = 5.50; // 灯条长度，单位：厘米
    constexpr double SMALL_ARMOR_WIDTH = 13.50; // 小装甲板宽度，单位：厘米
    constexpr double BIG_ARMOR_WIDTH = 23.0;   // 大装甲板宽度，单位：厘米
}

Solver::Solver( const SolverConfig& config ):
    cameraMatrix(config.camera_matrix),
    distCoeffs(config.distortion_coeffs),
    R_Cam_to_gripper(config.R_Cam_to_gripper),
    T_Cam_to_gripper(config.T_Cam_to_gripper),
    err_threshold(config.err_threshold)
{

    using namespace SASize;
    this->objectBigArmorP = {
        {-BIG_ARMOR_WIDTH / 2.0,  -LIGHTBAR_LENGTH / 2.0, 0},
        {BIG_ARMOR_WIDTH / 2.0,  -LIGHTBAR_LENGTH / 2.0, 0},
        {BIG_ARMOR_WIDTH / 2.0, LIGHTBAR_LENGTH / 2.0, 0},
        {-BIG_ARMOR_WIDTH / 2.0, LIGHTBAR_LENGTH / 2.0, 0}
    };

    this->objectSmallArmorP = {
        {-SMALL_ARMOR_WIDTH / 2.0,  -LIGHTBAR_LENGTH / 2.0, 0},
        {SMALL_ARMOR_WIDTH / 2.0,  -LIGHTBAR_LENGTH / 2.0, 0},
        {SMALL_ARMOR_WIDTH / 2.0, LIGHTBAR_LENGTH / 2.0, 0},
        {-SMALL_ARMOR_WIDTH / 2.0, LIGHTBAR_LENGTH / 2.0, 0}
    };
    
}

//解算单个装甲板的位置
std::array<ArmorPosi,2> Solver::operator () (const CVArmor& armor)
{
    //ArmorPosi(posi, face, toward, std::atan2(toward.z,toward.x), error);
    Eigen::Matrix<double, 3, 1> center_small, center_big;
    Eigen::Matrix<double, 3, 1> left_bottom_corner_small, left_bottom_corner_big;
    double error_small,error_big;
    double theta_small, theta_big;

    bool small_isInRange = true, big_isInRange = true;

    std::vector<cv::Mat> rvecs,tvecs;
    std::vector<double> reprojectionError;
    Eigen::Matrix<double, 1, 3> Z_inCamera(0,0,1);//沿着z轴的方向
    //当做小装甲板解算
    {
        int solutions = cv::solvePnPGeneric(
            this->objectSmallArmorP,
            armor.Lightcorners,
            cameraMatrix,
            distCoeffs,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE, // 使用 IPPE 算法获取多个解
            cv::noArray(),
            cv::noArray(),
            reprojectionError
        );

        // if(reprojectionError[0]>10||reprojectionError[1]) continue;
        // std::cerr<<reprojectionError.front()<<" "<<reprojectionError.back()<<std::endl;
        //筛选歧义解
        size_t valid_solution_idx = reprojectionError.front() < reprojectionError.back() ? 0 : reprojectionError.size()-1;

        center_small = Eigen::Map<Eigen::Vector3d>(tvecs[valid_solution_idx].ptr<double>());

        // 直接映射指针！
        Eigen::Map<Eigen::Vector3d> eigen_rvec(rvecs[valid_solution_idx].ptr<double>());
        
        // 将旋转向量转换为旋转矩阵
        // 提取旋转角度（模长）
        double angle = eigen_rvec.norm();
        Eigen::Matrix3d R;

        // (加一个极小值判断，防止目标完全没旋转时归一化除以 0)
        if (angle < 1e-6) {
            R = Eigen::Matrix3d::Identity();
        } else {
            Eigen::Vector3d axis = eigen_rvec.normalized(); // 提取归一化的旋转轴
            R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }
        Eigen::Matrix<double, 3, 1> left_bottom_corner_inArmor = 
            Eigen::Matrix<double, 3, 1>(-SASize::SMALL_ARMOR_WIDTH / 2.0, -SASize::LIGHTBAR_LENGTH / 2.0, 0);

        left_bottom_corner_small = R * left_bottom_corner_inArmor + center_small;

        //求解theta_small
        double cos_theta = std::abs(Z_inCamera * R.block<3,1>(0,0));
        theta_small = std::acos(std::clamp(cos_theta, 0.0, 1.0));

        //判断是否在有效范围内

        //检查重投影
        if (reprojectionError[valid_solution_idx] > this->err_threshold) small_isInRange = false;

        // 检查距离
        if (center_small.norm() > this->camera_range.distance_max) small_isInRange = false;

        // 计算偏航角、俯仰角和滚转角
        //pitch:
        double pitch = std::asin(std::clamp(R(1,2), -1.0, 1.0));
        if(pitch > this->camera_range.pitch_max || pitch < this->camera_range.pitch_min) small_isInRange = false;

        //yaw:
        double yaw = std::atan2( R(0,2), R(2,2) );//posi.face.x, posi.face.z;
        yaw = std::abs(yaw);
        if(yaw > this->camera_range.yaw_max) small_isInRange = false;

        //roll:
        double roll = std::asin(std::clamp(R(1,0), -1.0, 1.0));
        roll = std::abs(roll);
        if(roll > this->camera_range.roll_max) small_isInRange = false;
    }

    //当做大装甲板计算
    {
        int solutions = cv::solvePnPGeneric(
            this->objectBigArmorP,
            armor.Lightcorners,
            cameraMatrix,
            distCoeffs,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE, // 使用 IPPE 算法获取多个解
            cv::noArray(),
            cv::noArray(),
            reprojectionError
        );

        // if(reprojectionError[0]>10||reprojectionError[1]) continue;
        // std::cerr<<reprojectionError.front()<<" "<<reprojectionError.back()<<std::endl;
        //筛选歧义解
        size_t valid_solution_idx = reprojectionError.front() < reprojectionError.back() ? 0 : reprojectionError.size()-1;

        center_big = Eigen::Map<Eigen::Vector3d>(tvecs[valid_solution_idx].ptr<double>());

        // 直接映射指针！
        Eigen::Map<Eigen::Vector3d> eigen_rvec(rvecs[valid_solution_idx].ptr<double>());
        
        // 将旋转向量转换为旋转矩阵
        // 提取旋转角度（模长）
        double angle = eigen_rvec.norm();
        Eigen::Matrix3d R;

        // (加一个极小值判断，防止目标完全没旋转时归一化除以 0)
        if (angle < 1e-6) {
            R = Eigen::Matrix3d::Identity();
        } else {
            Eigen::Vector3d axis = eigen_rvec.normalized(); // 提取归一化的旋转轴
            R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }
        Eigen::Matrix<double, 3, 1> left_bottom_corner_inArmor = 
            Eigen::Matrix<double, 3, 1>(-SASize::BIG_ARMOR_WIDTH / 2.0, -SASize::LIGHTBAR_LENGTH / 2.0, 0);

        left_bottom_corner_big = R * left_bottom_corner_inArmor + center_big;

        //求解theta_big
        double cos_theta = std::abs(Z_inCamera * R.block<3,1>(0,0));
        theta_big = std::acos(std::clamp(cos_theta, 0.0, 1.0));

        //判断是否在有效范围内

        //检查重投影
        if (reprojectionError[valid_solution_idx] > this->err_threshold) big_isInRange = false;

        // 检查距离
        if (center_big.norm() > this->camera_range.distance_max) big_isInRange = false;

        // 计算偏航角、俯仰角和滚转角
        //pitch:
        double pitch = std::asin(std::clamp(R(1,2), -1.0, 1.0));
        if(pitch > this->camera_range.pitch_max || pitch < this->camera_range.pitch_min) big_isInRange = false;

        //yaw:
        double yaw = std::atan2( R(0,2), R(2,2) );//posi.face.x, posi.face.z;
        yaw = std::abs(yaw);
        if(yaw > this->camera_range.yaw_max) big_isInRange = false;

        //roll:
        double roll = std::asin(std::clamp(R(1,0), -1.0, 1.0));
        roll = std::abs(roll);
        if(roll > this->camera_range.roll_max) big_isInRange = false;

    }
    
    return std::array<ArmorPosi, 2> { 
        ArmorPosi{center_small, left_bottom_corner_small, theta_small, error_small, small_isInRange}, 
        ArmorPosi{center_big, left_bottom_corner_big, theta_big, error_big, big_isInRange}
    };
}

std::vector<ArmorPosi> Solver::operator()(const std::vector<YoloArmor>& armors)
{
    std::vector<ArmorPosi> results;
    if (armors.empty()) return results;
    results.reserve(armors.size());
    Eigen::Matrix<double, 1, 3> Z_inCamera(0,0,1);//沿着z轴的方向

    for (const auto& yolo_armor : armors) {
        // 1. 根据 class_id 确定类型和 3D 模型
        ArmorPosi::Type target_type;
        bool is_big = false;
        int id = yolo_armor.class_id;

        // 映射逻辑: 只有 base 和 hero 是大装甲板
        switch (id) {
            // 哨兵 (0-2)
            case 0: case 1: case 2:
                target_type = ArmorPosi::Type::guard;
                is_big = false;
                break;

            // 英雄 (3-5) - 大装甲板
            case 3: case 4: case 5:
                target_type = ArmorPosi::Type::hero;
                is_big = true;
                break;

            // 2号步兵 (6-8)
            case 6: case 7: case 8:
                target_type = ArmorPosi::Type::two;
                is_big = false;
                break;

            // 3号步兵 (9-11)
            case 9: case 10: case 11:
                target_type = ArmorPosi::Type::three;
                is_big = false;
                break;

            // 4号步兵 (12-14)
            case 12: case 13: case 14:
                target_type = ArmorPosi::Type::four;
                is_big = false;
                break;

            // 前哨站 (18-20)
            case 18: case 19: case 20:
                target_type = ArmorPosi::Type::outpost;
                is_big = false;
                break;

            // 大基地 (21-24) - 大装甲板
            case 21: case 22: case 23: case 24:
                target_type = ArmorPosi::Type::base;
                is_big = true;
                break;

            // 小基地 (25-28)
            case 25: case 26: case 27: case 28:
                target_type = ArmorPosi::Type::base;
                is_big = false;
                break;

            // 未知类型，跳过
            default:
                continue;
        }

        // 选择对应的 3D 物体坐标系参考点
        const auto& objectPoints = is_big ? this->objectBigArmorP : this->objectSmallArmorP;

        // 2. 直接进行 PnP 解算 (参考第一个函数的解算逻辑)
        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<double> reprojectionError;

        cv::solvePnPGeneric(
            objectPoints,
            yolo_armor.keypoints,
            this->cameraMatrix,
            this->distCoeffs,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE,
            cv::noArray(),
            cv::noArray(),
            reprojectionError
        );

        if (rvecs.empty()) continue;

        size_t valid_solution_idx = reprojectionError.front() < reprojectionError.back() ? 0 : reprojectionError.size()-1;

        Eigen::Matrix<double, 3, 1> center = Eigen::Map<Eigen::Vector3d>(tvecs[valid_solution_idx].ptr<double>());

        // 直接映射指针！
        Eigen::Map<Eigen::Vector3d> eigen_rvec(rvecs[valid_solution_idx].ptr<double>());
        
        // 将旋转向量转换为旋转矩阵
        // 提取旋转角度（模长）
        double angle = eigen_rvec.norm();
        Eigen::Matrix3d R;

        // (加一个极小值判断，防止目标完全没旋转时归一化除以 0)
        if (angle < 1e-6) {
            R = Eigen::Matrix3d::Identity();
        } else {
            Eigen::Vector3d axis = eigen_rvec.normalized(); // 提取归一化的旋转轴
            R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }


        // 4. 计算结果并填充 ArmorPosi
        Eigen::Matrix<double, 3, 1> left_bottom_corner_inArmor = is_big ?
            Eigen::Matrix<double, 3, 1>(-SASize::BIG_ARMOR_WIDTH / 2.0, -SASize::LIGHTBAR_LENGTH / 2.0, 0):
            Eigen::Matrix<double, 3, 1>(-SASize::SMALL_ARMOR_WIDTH / 2.0, -SASize::LIGHTBAR_LENGTH / 2.0, 0);

        Eigen::Matrix<double, 3, 1> left_bottom_corner = R * left_bottom_corner_inArmor + center;

        //求解theta_big
        double cos_theta = std::abs(Z_inCamera * R.block<3,1>(0,0));
        double theta = std::acos(std::clamp(cos_theta, 0.0, 1.0));

        //判断是否在有效范围内
        bool isInRange = true;

        //检查重投影
        if (reprojectionError[valid_solution_idx] > this->err_threshold) isInRange = false;

        // 检查距离
        if (center.norm() > this->camera_range.distance_max) isInRange = false;

        // 计算偏航角、俯仰角和滚转角
        //pitch:
        double pitch = std::asin(std::clamp(R(1,2), -1.0, 1.0));
        if(pitch > this->camera_range.pitch_max || pitch < this->camera_range.pitch_min) isInRange = false;

        //yaw:
        double yaw = std::atan2( R(0,2), R(2,2) );
        yaw = std::abs(yaw);
        if(yaw > this->camera_range.yaw_max) isInRange = false;

        //roll:
        double roll = std::asin(std::clamp(R(1,0), -1.0, 1.0));
        roll = std::abs(roll);
        if(roll > this->camera_range.roll_max) isInRange = false;

        if(!isInRange) continue;
        // 5. 构造并存入结果
        results.emplace_back(center, left_bottom_corner, theta, reprojectionError[valid_solution_idx], isInRange);
        results.back().type = target_type;
        results.back().confidence = yolo_armor.conf;
    }

    return results;
}

std::vector<ArmorPosi> Solver::operator()(const std::vector<YoloArmor>& armors, const Eigen::Quaterniond& gripper_to_world)
{
    std::vector<ArmorPosi> results;
    if (armors.empty()) return results;
    results.reserve(armors.size());
    Eigen::Matrix<double, 1, 3> Z_inCamera(0,0,1);//沿着z轴的方向

    for (const auto& yolo_armor : armors) {
        // 1. 根据 class_id 确定类型和 3D 模型
        ArmorPosi::Type target_type;
        bool is_big = false;
        int id = yolo_armor.class_id;

        // 映射逻辑: 只有 base 和 hero 是大装甲板
        switch (id) {
            // 哨兵 (0-2)
            case 0: case 1: case 2:
                target_type = ArmorPosi::Type::guard;
                is_big = false;
                break;

            // 英雄 (3-5) - 大装甲板
            case 3: case 4: case 5:
                target_type = ArmorPosi::Type::hero;
                is_big = true;
                break;

            // 2号步兵 (6-8)
            case 6: case 7: case 8:
                target_type = ArmorPosi::Type::two;
                is_big = false;
                break;

            // 3号步兵 (9-11)
            case 9: case 10: case 11:
                target_type = ArmorPosi::Type::three;
                is_big = false;
                break;

            // 4号步兵 (12-14)
            case 12: case 13: case 14:
                target_type = ArmorPosi::Type::four;
                is_big = false;
                break;

            // 前哨站 (18-20)
            case 18: case 19: case 20:
                target_type = ArmorPosi::Type::outpost;
                is_big = false;
                break;

            // 大基地 (21-24) - 大装甲板
            case 21: case 22: case 23: case 24:
                target_type = ArmorPosi::Type::base;
                is_big = true;
                break;

            // 小基地 (25-28)
            case 25: case 26: case 27: case 28:
                target_type = ArmorPosi::Type::base;
                is_big = false;
                break;

            // 未知类型，跳过
            default:
                continue;
        }

        // 选择对应的 3D 物体坐标系参考点
        const auto& objectPoints = is_big ? this->objectBigArmorP : this->objectSmallArmorP;

        // 2. 直接进行 PnP 解算 (参考第一个函数的解算逻辑)
        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<double> reprojectionError;

        cv::solvePnPGeneric(
            objectPoints,
            yolo_armor.keypoints,
            this->cameraMatrix,
            this->distCoeffs,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE,
            cv::noArray(),
            cv::noArray(),
            reprojectionError
        );

        if (rvecs.empty()) continue;

        size_t valid_solution_idx = reprojectionError.front() < reprojectionError.back() ? 0 : reprojectionError.size()-1;

        Eigen::Matrix<double, 3, 1> center = Eigen::Map<Eigen::Vector3d>(tvecs[valid_solution_idx].ptr<double>());

        // 直接映射指针！
        Eigen::Map<Eigen::Vector3d> eigen_rvec(rvecs[valid_solution_idx].ptr<double>());
        
        // 将旋转向量转换为旋转矩阵
        // 提取旋转角度（模长）
        double angle = eigen_rvec.norm();
        Eigen::Matrix3d R;

        // (加一个极小值判断，防止目标完全没旋转时归一化除以 0)
        if (angle < 1e-6) {
            R = Eigen::Matrix3d::Identity();
        } else {
            Eigen::Vector3d axis = eigen_rvec.normalized(); // 提取归一化的旋转轴
            R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }


        // 4. 计算结果并填充 ArmorPosi
        Eigen::Matrix<double, 3, 1> left_bottom_corner_inArmor = is_big ?
            Eigen::Matrix<double, 3, 1>(-SASize::BIG_ARMOR_WIDTH / 2.0, -SASize::LIGHTBAR_LENGTH / 2.0, 0):
            Eigen::Matrix<double, 3, 1>(-SASize::SMALL_ARMOR_WIDTH / 2.0, -SASize::LIGHTBAR_LENGTH / 2.0, 0);

        Eigen::Matrix<double, 3, 1> left_bottom_corner = R * left_bottom_corner_inArmor + center;

        //求解theta_big
        double cos_theta = std::abs(Z_inCamera * R.block<3,1>(0,0));
        double theta = std::acos(std::clamp(cos_theta, 0.0, 1.0));

        //判断是否在有效范围内
        bool isInRange = true;

        //检查重投影
        if (reprojectionError[valid_solution_idx] > this->err_threshold) isInRange = false;

        // 检查距离
        if (center.norm() > this->camera_range.distance_max) isInRange = false;
        
        {
            // 计算偏航角、俯仰角和滚转角
            //pitch:
            double pitch = std::asin(std::clamp(R(1,2), -1.0, 1.0));
            if(pitch > this->camera_range.pitch_max || pitch < this->camera_range.pitch_min) isInRange = false;

            //yaw:
            double yaw = std::atan2( R(0,2), R(2,2) );
            yaw = std::abs(yaw);
            if(yaw > this->camera_range.yaw_max) isInRange = false;

            //roll:
            double roll = std::asin(std::clamp(R(1,0), -1.0, 1.0));
            roll = std::abs(roll);
            if(roll > this->camera_range.roll_max) isInRange = false;
        }

        if(!isInRange) continue;

        //转换到世界坐标系
        Eigen::Matrix3d RGtoW = gripper_to_world.toRotationMatrix();
        Eigen::Matrix3d R_Cam_to_World = RGtoW * this->R_Cam_to_gripper;
        Eigen::Matrix<double, 3, 1> T_Cam_to_World = RGtoW * this->T_Cam_to_gripper;

        //将装甲板坐标系转换到世界坐标系
        center = R_Cam_to_World * center + T_Cam_to_World;
        left_bottom_corner = R_Cam_to_World * left_bottom_corner + T_Cam_to_World;

        {
            Eigen::Matrix3d R_Armor_inWorld = R_Cam_to_World * R;
            
            //检查高度
            if (center(2) > this->world_range.high_max || center(2) < this->world_range.high_min) isInRange = false;

            //检查俯仰角
            double pitch = std::asin(std::clamp(R_Armor_inWorld(2,2), -1.0, 1.0));
            if (pitch > this->world_range.pitch_max || pitch < this->world_range.pitch_min) isInRange = false;

            //roll:
            double roll = std::abs(std::asin(std::clamp(R_Armor_inWorld(2,0), -1.0, 1.0)));
            if(roll > this->world_range.roll_max) isInRange = false;
        }

        if(!isInRange) continue;
        // 5. 构造并存入结果
        results.emplace_back(center, left_bottom_corner, theta, reprojectionError[valid_solution_idx], isInRange);
        results.back().type = target_type;
        results.back().confidence = yolo_armor.conf;
    }

    return results;
}

std::vector< std::array< ArmorPosi, 2> > Solver::operator()(const std::deque<CVArmor>& armors)
{
    std::vector< std::array< ArmorPosi, 2> > armors_posi;
    if(armors.empty()) return armors_posi;
    armors_posi.reserve(armors.size());

    for(const auto& armor:armors)
    {
        armors_posi.push_back(this->operator()(armor));//记录
    }
    return armors_posi;
}

std::vector< std::array< ArmorPosi, 2> > Solver::operator()(const std::vector<CVArmor>& armors)
{
    std::vector< std::array< ArmorPosi, 2> > armors_posi;
    if(armors.empty()) return armors_posi;
    armors_posi.reserve(armors.size());

    for(const auto& armor:armors)
    {
        armors_posi.push_back(this->operator()(armor));//记录
    }
    return armors_posi;
}




void Solver::ConverToWorld(std::array<ArmorPosi,2>& armor_posis, const Eigen::Quaterniond& gripper_to_world)
{
    Eigen::Matrix<double, 3, 3> RGtoW = gripper_to_world.toRotationMatrix();//手坐标系到世界坐标系的旋转矩阵
    Eigen::Matrix3d R = RGtoW * this->R_Cam_to_gripper;// 相机坐标系到世界坐标系的旋转矩阵
    Eigen::Matrix<double, 3, 1> T = RGtoW * this->T_Cam_to_gripper;// 相机坐标系到世界坐标系的平移向量

    armor_posis[0].center = R * armor_posis[0].center + T;
    armor_posis[0].left_bottom_corner = R * armor_posis[0].left_bottom_corner + T;

    armor_posis[1].center = R * armor_posis[1].center + T;
    armor_posis[1].left_bottom_corner = R * armor_posis[1].left_bottom_corner + T;
}

void Solver::ConverToWorld(std::vector< std::array<ArmorPosi,2> >& armors_posis, const Eigen::Quaterniond& gripper_to_world)
{
    Eigen::Matrix<double, 3, 3> RGtoW = gripper_to_world.toRotationMatrix();
    Eigen::Matrix3d R = RGtoW * this->R_Cam_to_gripper;// 手坐标系到世界坐标系的旋转矩阵
    Eigen::Matrix<double, 3, 1> T = RGtoW * this->T_Cam_to_gripper;// 手坐标系到世界坐标系的平移向量

    for(auto& armor_posis:armors_posis)
    {
        armor_posis[0].center = R * armor_posis[0].center + T;
        armor_posis[0].left_bottom_corner = R * armor_posis[0].left_bottom_corner + T;
        armor_posis[1].center = R * armor_posis[1].center + T;
        armor_posis[1].left_bottom_corner = R * armor_posis[1].left_bottom_corner + T;
    }
}


void Solver::ConverToWorld(ArmorPosi& armor_posi, const Eigen::Quaterniond& gripper_to_world)
{
    Eigen::Matrix<double, 3, 3> RGtoW = gripper_to_world.toRotationMatrix();//手坐标系到世界坐标系的旋转矩阵
    Eigen::Matrix3d R = RGtoW * this->R_Cam_to_gripper;// 相机坐标系到世界坐标系的旋转矩阵
    Eigen::Matrix<double, 3, 1> T = RGtoW * this->T_Cam_to_gripper;// 相机坐标系到世界坐标系的平移向量

    armor_posi.center = R * armor_posi.center + T;
    armor_posi.left_bottom_corner = R * armor_posi.left_bottom_corner + T;
}

void Solver::ConverToWorld(std::vector<ArmorPosi>& armors_posi, const Eigen::Quaterniond& gripper_to_world)
{
    Eigen::Matrix<double, 3, 3> RGtoW = gripper_to_world.toRotationMatrix();// 手坐标系到世界坐标系的旋转矩阵
    Eigen::Matrix3d R = RGtoW * this->R_Cam_to_gripper;// 相机坐标系到世界坐标系的旋转矩阵
    Eigen::Matrix<double, 3, 1> T = RGtoW * this->T_Cam_to_gripper;// 相机坐标系到世界坐标系的平移向量

    for(auto& armor_posi:armors_posi)
    {
        armor_posi.center = R * armor_posi.center + T;
        armor_posi.left_bottom_corner = R * armor_posi.left_bottom_corner + T;
    }
}


void Solver::ansShow(const Eigen::Matrix<double, 3, 1>& posi,cv::Mat& image)
{
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); // 单位旋转向量
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F); // 单位平移向量

    // 3. 执行投影
    // cv::projectPoints 需要一个点的向量作为输入
    std::vector<cv::Point3d> objectPoints;
    objectPoints.push_back(cv::Point3d(posi(0), posi(1), posi(2)));

    // 用于存储投影结果的2D点向量
    std::vector<cv::Point2d> imagePoints;

    //重投影
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    //在图像上绘制结果
    // 输出和可视化结果
    // 投影后的2D点坐标
    cv::Point2d projectedPoint = imagePoints[0];
    int imageWidth = image.cols;
    int imageHeight = image.rows;

    // 在图像上绘制投影点 (画一个红色的圆圈)
    // 检查点是否在图像范围内
    if (projectedPoint.x >= 0 && projectedPoint.x < imageWidth &&
        projectedPoint.y >= 0 && projectedPoint.y < imageHeight)
    {
        cv::circle(image, projectedPoint, 5, cv::Scalar(0, 0, 255), -1); // 红色实心圆
        cv::putText(image, "Projected Point", cv::Point(projectedPoint.x + 10, projectedPoint.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    } else {
        std::cout << "Projected point is outside the image frame." << std::endl;
    }
    // 显示图像
    // cv::imshow("Projected Point Visualization", image);
    // cv::waitKey(1); // 等待按键后退出
}

// void Solver::Filter(std::vector< std::array<ArmorPosi,2> >& armors_posis,
//                     const Eigen::Quaterniond& gripper_to_world,
//                     const Eigen::Matrix<double, 3, 1>& Gun,
//                     const size_t num)
// {
//     if (armors_posis.size() <= num) return;


//     //选择与枪管夹角最小的num个装甲板
//     // 1. 初始化索引数组 [0, 1, 2, ..., n-1]
//     std::vector<size_t> indices(armors_posis.size());
//     std::iota(indices.begin(), indices.end(), 0);

//         // 3. 按夹角部分排序（找出夹角最小的 num 个）
//         std::partial_sort(indices.begin(), indices.begin() + num, indices.end(),
//             [&](size_t i1, size_t i2) {
//                 // 取重投影误差小的装甲板的坐标作为代表进行比较


//                 const auto& p1 = armors_posis[i1][0].center;
//                 const auto& p2 = armors_posis[i2][0].center;

//                 // 组装为 Eigen 向量并归一化
//                 Eigen::Vector3d v1(p1.x, p1.y, p1.z);
//                 Eigen::Vector3d v2(p2.x, p2.y, p2.z);
//                 v1.normalize();
//                 v2.normalize();

//                 // 比较余弦值（点乘结果）。cos值越大，说明夹角越小
//                 return v1.dot(gun_vec) > v2.dot(gun_vec); 
//             });

//         // 4. 根据排序好的索引提取结果
//         ArmorPosi dummy_armor(cv::Point3d(0,0,0), cv::Point3d(0,0,0), cv::Point3d(0,0,0), 0.0, 0.0);

//         // 创建一个包含两个 dummy_armor 的默认 array
//         std::array<ArmorPosi, 2> default_array = {dummy_armor, dummy_armor};
//         armors_posis.resize(num,default_array);
//         armors_pattern.resize(num);
//         for (int i = 0; i < num; ++i) {
//             armors_posis[i] = armors_posis_result[indices[i]];
//             armors_pattern[i] = armors_pattern_result[indices[i]];
//         }

//         return;
//     }

// }

// void Solver::ansShow(const ArmorPosi& armor,cv::Mat& image)
// {
//     double high = 27.5, width;
//     if(armor.type == ArmorPosi::Type::hero || armor.type == ArmorPosi::Type::base)
//         width = 115.0;
//     else
//         width = 67.5;

//     cv::Point3d toward_w = width * armor.toward;
//     cv::Point3d toward_h = high * (armor.face.cross(armor.toward)/cv::norm(armor.face.cross(armor.toward)));

//     cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); // 单位旋转向量
//     cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F); // 单位平移向量

//     // 执行投影
//     // cv::projectPoints 需要一个点的向量作为输入
//     std::vector<cv::Point3d> objectPoints;
//     objectPoints.reserve(5);
//     objectPoints.push_back(armor.posi);
//     objectPoints.push_back(armor.posi - toward_w - toward_h);
//     objectPoints.push_back(armor.posi + toward_w - toward_h);
//     objectPoints.push_back(armor.posi + toward_w + toward_h);
//     objectPoints.push_back(armor.posi - toward_w + toward_h);


//     // 用于存储投影结果的2D点向量
//     std::vector<cv::Point2d> imagePoints;
//     imagePoints.reserve(5);

//     //重投影
//     cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

//     //在图像上绘制结果
//     cv::Point2d CenterPoint = imagePoints[0];
//     int imageWidth = image.cols;
//     int imageHeight = image.rows;

//     // 在图像上绘制投影点 (画一个红色的圆圈)
//     // 检查点是否在图像范围内
//     if (CenterPoint.x >= 0 && CenterPoint.x < imageWidth &&
//         CenterPoint.y >= 0 && CenterPoint.y < imageHeight)
//     {
//         cv::circle(image, CenterPoint, 5, cv::Scalar(0, 0, 255), -1); // 红色实心圆
//         cv::putText(image, "Projected Point", cv::Point(CenterPoint.x + 10, CenterPoint.y),
//                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
//     } else {
//         std::cout << "Projected point is outside the image frame." << std::endl;
//     }

//     //绘制装甲板轮廓
//     std::vector<cv::Point2d> points;
//     points.reserve(4);
//     for(int i=1;i<=4;i++)
//     {
//         cv::Point2d Point = imagePoints[i];
//         if (Point.x >= 0 && Point.x < imageWidth &&
//         Point.y >= 0 && Point.y < imageHeight)
//         {
//             points.push_back(Point);
//         } else {
//             std::cout << "Projected point is outside the image frame." << std::endl;
//             return;
//         }
//     }

//     // 绘制
//     std::vector<std::vector<cv::Point2d>> contours{points};
//     cv::polylines(image,contours,1,cv::Scalar(0, 255, 0),3,cv::LINE_AA);
// }

// void Solver::FilterAndConverToWorld(std::vector<ArmorPosi>& armors_posi,
//                     const Eigen::Quaterniond& gripper_to_world,
//                     const Eigen::Matrix<double, 3, 1>& Gun,
//                     const size_t num)
// {
//     std::vector<ArmorPosi> armors_posi_result;
//     armors_posi_result.reserve(armors_posi.size());

//     for (auto& armor_posi : armors_posi)
//     {
//         // 解算误差筛选
//         if (armor_posi.error > 1) continue;

//         // 相机系下的距离筛选
//         if (cv::norm(armor_posi.posi) > 800) continue;

//         // 坐标系变换到世界坐标系
//         this->ConverToWorld(armor_posi, gripper_to_world);

//         // 高度筛选
//         if (armor_posi.posi.z > 2000 || armor_posi.posi.z < -50) continue;

//         // 角度筛选
//         const auto& face = armor_posi.toward.cross(armor_posi.face);
//         cv::Point3d base{armor_posi.posi.x, armor_posi.posi.y, 0};
//         base = base / cv::norm(base);
//         double angle = base.dot(face);

//         if (angle < -0.5 || angle > 0.85) continue;

//         // 储存筛选结果
//         armors_posi_result.emplace_back(armor_posi);
//     }

//     // 选择与枪管夹角最小的num个装甲板
//     if (armors_posi_result.size() > num)
//     {
//         // 初始化索引数组
//         std::vector<size_t> indices(armors_posi_result.size());
//         std::iota(indices.begin(), indices.end(), 0);

//         // 枪管方向归一化
//         Eigen::Vector3d gun_vec = Gun.normalized();

//         // 按夹角部分排序
//         std::partial_sort(indices.begin(), indices.begin() + num, indices.end(),
//             [&](size_t i1, size_t i2) {
//                 const auto& p1 = armors_posi_result[i1].posi;
//                 const auto& p2 = armors_posi_result[i2].posi;

//                 Eigen::Vector3d v1(p1.x, p1.y, p1.z);
//                 Eigen::Vector3d v2(p2.x, p2.y, p2.z);
//                 v1.normalize();
//                 v2.normalize();

//                 return v1.dot(gun_vec) > v2.dot(gun_vec);
//             });

//         // 根据排序好的索引提取结果
//         // 4. 根据排序好的索引提取结果
//         ArmorPosi dummy_armor(cv::Point3d(0,0,0), cv::Point3d(0,0,0), cv::Point3d(0,0,0), 0.0, 0.0);

//         armors_posi.resize(num, dummy_armor);
//         for (size_t i = 0; i < num; ++i) {
//             armors_posi[i] = armors_posi_result[indices[i]];
//         }
//         return;
//     }

//     // 更新装甲板位置
//     armors_posi = std::move(armors_posi_result);
// }

