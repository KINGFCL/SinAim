#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <glob.h>

using namespace cv;
using namespace std;
cv::Point3d rotationMatrixToEulerAngles(const cv::Mat &R);

//pitch轴是否有误差
// #define R_error 
// const double R_err = 14; //mm

int main()
{
    // 棋盘参数设置
    const Size BOARD_SIZE(11, 8);  // 棋盘内角点数量 (列数, 行数)
    const double SQUARE_SIZE = 2.0;  // 每个方格的实际尺寸 (厘米)
    
    string config_path = "../Data/Calibration_R_T.yaml";
    string image_path = "../Data/images/*.png";
    
    //
    //加载存储数据的YAML文件
    FileStorage fs;
    if (!fs.open(config_path, FileStorage::READ)) {
        cerr << "Error: Failed to open YAML file: " << config_path << endl;
        return -1;  // 失败时返回
    } else {
        cout << "open YAML yes" << endl;
    }


    // 创建棋盘的世界坐标系坐标点
    vector<Point3f> objectPoints;
    for (int i = 0; i < BOARD_SIZE.height; i++) {
        for (int j = 0; j < BOARD_SIZE.width; j++) {
            objectPoints.push_back(Point3f(j * SQUARE_SIZE, i * SQUARE_SIZE, 0));
        }
    }
    
    // 存储所有图像的角点坐标和对应的世界坐标
    vector<vector<Point2f>> imagePointsAll;
    vector<vector<Point3f>> objectPointsAll;
    
    // 获取图像文件列表
    vector<String> imageFiles;
    vector<String> havChessBFiles;
    glob(image_path, imageFiles);
    
    if (imageFiles.empty()) {
        cout << "错误: 在images文件夹中没有找到PNG图片文件!" << endl;
        return -1;
    }
    
    cout << "找到 " << imageFiles.size() << " 张图片" << endl;
    
    Size imageSize;
    int validImages = 0;
    
    // 处理每张图片
    for (size_t i = 0; i < imageFiles.size(); i++) {
        Mat image = imread(imageFiles[i]);
        if (image.empty()) {
            cout << "无法读取图片: " << imageFiles[i] << endl;
            continue;
        }
        
        // 转换为灰度图
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        imshow("ia",gray);
        waitKey(0);
        
        if (imageSize.width == 0) {
            imageSize = gray.size();
        }
        
        // 查找棋盘角点
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, BOARD_SIZE, corners,
                                         CALIB_CB_ADAPTIVE_THRESH | 
                                         CALIB_CB_NORMALIZE_IMAGE |
                                         CALIB_CB_FAST_CHECK);
        
        if (found) {
            // 亚像素精确化角点位置
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                        TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            
            // 存储角点
            imagePointsAll.push_back(corners);
            objectPointsAll.push_back(objectPoints);

            //记录图片的名字
            havChessBFiles.push_back(imageFiles[i]);
            validImages++;
            
            // 绘制角点（可选，用于验证）
            drawChessboardCorners(image, BOARD_SIZE, corners, found);
            
            cout << "图片 " << i + 1 << "/" << imageFiles.size() 
                 << " 处理成功: " << imageFiles[i] << endl;
            
            // 显示结果（可选）
            Mat resized;
            resize(image, resized, Size(800, 600));
            imshow("棋盘角点检测", resized);
            waitKey(100);  // 短暂显示
        } else {
            cout << "图片 " << i + 1 << "/" << imageFiles.size() 
                 << " 未找到棋盘: " << imageFiles[i] << endl;
        }
    }
    
    destroyAllWindows();
    
    if (validImages < 3) {
        cout << "错误: 需要至少3张有效的棋盘图片进行标定，当前只有 " 
             << validImages << " 张" << endl;
        return -1;
    }
    
    cout << "\n开始相机标定，使用 " << validImages << " 张有效图片..." << endl;
    
    // 相机标定
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
    vector<Mat> rvecs, tvecs;
    
    double rms = calibrateCamera(objectPointsAll, imagePointsAll, imageSize,
                                cameraMatrix, distCoeffs, rvecs, tvecs);
    
    cout << "\n=== 相机标定结果 ===" << endl;
    cout << "RMS重投影误差: " << rms << " 像素" << endl;
    cout << "\n内参矩阵 (Camera Matrix):" << endl;
    cout << cameraMatrix << endl;
    cout << "\n畸变系数 (Distortion Coefficients):" << endl;
    cout << distCoeffs << endl;
    
    // // 保存标定结果到文件
    // FileStorage fs("camera_calibration.yml", FileStorage::WRITE);
    // fs << "camera_matrix" << cameraMatrix;
    // fs << "distortion_coefficients" << distCoeffs;
    // fs << "image_width" << imageSize.width;
    // fs << "image_height" << imageSize.height;
    // fs << "rms_error" << rms;
    // fs << "valid_images" << validImages;
    // fs.release();
    
    // cout << "\n标定结果已保存到 camera_calibration.yml" << endl;
    
    // 计算标定精度评估
    vector<float> perViewErrors;
    double totalError = 0;
    
    for (size_t i = 0; i < objectPointsAll.size(); i++) {
        vector<Point2f> projectedPoints;
        projectPoints(objectPointsAll[i], rvecs[i], tvecs[i], 
                     cameraMatrix, distCoeffs, projectedPoints);
        
        double error = norm(imagePointsAll[i], projectedPoints, NORM_L2);
        perViewErrors.push_back((float)(error / objectPointsAll[i].size()));
        totalError += error * error;
    }
    
    double meanError = sqrt(totalError / (validImages * BOARD_SIZE.width * BOARD_SIZE.height));
    cout << "平均重投影误差: " << meanError << " 像素" << endl;
    
    cout << "\n=== 标定完成 ===" << endl;
    cout << "建议: RMS误差小于1.0像素表示标定质量良好" << endl;

    // 开始手眼标定
    vector<Mat> Rs_world_to_camera,Ts_world_to_camera;
    Ts_world_to_camera = tvecs;
    for (size_t i = 0; i < rvecs.size(); i++) {
        Mat R;
        Rodrigues(rvecs[i], R);
        Rs_world_to_camera.push_back(R);
    }

    vector<Mat> Rs_base_to_hand,Ts_base_to_hand;
    for(size_t i=0;i<havChessBFiles.size();i++)
    {
        if(havChessBFiles[i].length()<5)
        {
            cerr<<"错误: 文件名不合法: " << havChessBFiles[i] << endl;
            return 0;
        }

        string key = havChessBFiles[i].substr(15, havChessBFiles[i].length() );
        key = key.substr(0,key.length()-4);
        double tdata[3]={0.,0.,0.};
        Mat R,T(3,1,CV_64F,tdata);
        fs[key] >> R;
        std::cout<<R<< key<<"\n";
        Rs_base_to_hand.push_back(R);

        #ifdef R_error
        auto EulerAngles = rotationMatrixToEulerAngles(R);
        std::cout<<"EulerAngles: "<<EulerAngles.x<<" "<<EulerAngles.y<<" "<<EulerAngles.z<<"\n";
        double theta_pitch = EulerAngles.y;
        double theta_yaw = EulerAngles.x;

        double z_err = -( R_err * sin(theta_pitch) );
        double L_err = R_err * (1 - cos(theta_pitch) );
        double x_err = L_err * cos(theta_yaw);
        double y_err = L_err * sin(theta_yaw);
        T = (Mat_<double>(3,1) << x_err, y_err, z_err);
        std::cout<<"误差修正: "<<x_err<<" "<<y_err<<" "<<z_err<<"\n";
        #endif

        Ts_base_to_hand.push_back(T);
    }
    
    Mat R_hand_to_cam_out, T_hand_to_cam_out,
        R_base_to_world_out, T_base_to_world_out;

    calibrateRobotWorldHandEye(Rs_world_to_camera, Ts_world_to_camera,
                               Rs_base_to_hand, Ts_base_to_hand,
                               R_base_to_world_out, T_base_to_world_out,
                               R_hand_to_cam_out, T_hand_to_cam_out,
                               CALIB_ROBOT_WORLD_HAND_EYE_SHAH
                               );

    cout<<"----------------------------------"<<endl;
    cout<<"手眼标定完成："<<endl;

    // 1. 计算眼到手的旋转矩阵 (转置)
    cv::Mat R_cam_to_hand = R_hand_to_cam_out.t();

    // 2. 计算眼到手的平移向量 (-R^T * t)
    // 注意：这里必须用矩阵乘法，不能直接减
    cv::Mat T_cam_to_hand = -R_cam_to_hand * T_hand_to_cam_out;

    cout << "手眼标定完成：" << endl;
    cout << "手到眼的旋转矩阵：" << "\n" << R_hand_to_cam_out << endl;
    cout << "手到眼的平移向量：" << "\n" << T_hand_to_cam_out << endl;

    cout << "眼到手的旋转矩阵： " << "\n" << R_cam_to_hand << endl;
    cout << "眼到手的平移向量：" << "\n" << T_cam_to_hand << endl;


    fs.release(); // 关闭文件


    cout << "----------------------------------" << endl;
    cout << "手眼标定完成，结果已输出。" << endl;

    // --- 修正后的误差分析代码 ---
    cout << "正在计算手眼标定误差..." << endl;


    // 1. 构造标定出的 手->眼 变换矩阵 (T_Hand_to_Cam)
    Mat T_hand_to_cam = Mat::eye(4, 4, CV_64F);
    R_hand_to_cam_out.copyTo(T_hand_to_cam(Rect(0, 0, 3, 3)));
    T_hand_to_cam_out.copyTo(T_hand_to_cam(Rect(3, 0, 1, 3)));

    // 用于统计
    vector<Point3f> board_positions_in_base; // 标定板原点在基座系下的位置
    Point3f mean_position(0, 0, 0);
    vector<Mat> R_base_to_worlds; // 存储旋转以便计算角度误差

    for (size_t i = 0; i < Rs_world_to_camera.size(); i++) {
        // A. 构造 基座->手 (T_Base_to_Hand)
        // 云台模式下：T部分全为0 (或者非常小的偏移)，R随云台转动
        Mat T_base_to_hand_i = Mat::eye(4, 4, CV_64F);
        Rs_base_to_hand[i].copyTo(T_base_to_hand_i(Rect(0, 0, 3, 3)));
        // 注意：这里用你原始的 Ts_base_to_hand[i] (即 0,0,0)
        Ts_base_to_hand[i].copyTo(T_base_to_hand_i(Rect(3, 0, 1, 3))); 

        // B. 构造 世界->相机 (T_World_to_Cam) -> 对应标定板数据
        Mat T_world_to_cam_i = Mat::eye(4, 4, CV_64F);
        Mat R_w2c;
        Rodrigues(rvecs[i], R_w2c);
        R_w2c.copyTo(T_world_to_cam_i(Rect(0, 0, 3, 3)));
        tvecs[i].copyTo(T_world_to_cam_i(Rect(3, 0, 1, 3)));

        // C. 计算链：基座 -> 手 -> 相机 -> 世界 (标定板)
        // 公式：T_Base_to_World = T_Base_to_Hand * T_Hand_to_Cam * T_Cam_to_World
        // 其中 T_Cam_to_World 是 T_World_to_Cam 的逆矩阵
        Mat T_cam_to_world_i = T_world_to_cam_i.inv();
        
        Mat T_base_to_world_calc = T_base_to_hand_i * T_hand_to_cam * T_cam_to_world_i;

        // D. 提取计算出的标定板位置 (平移部分)
        Mat pos_mat = T_base_to_world_calc(Rect(3, 0, 1, 3));
        Point3f pos(pos_mat.at<double>(0), pos_mat.at<double>(1), pos_mat.at<double>(2));
        
        board_positions_in_base.push_back(pos);
        mean_position += pos;
        
        // 存储旋转部分用于后续角度误差分析
        R_base_to_worlds.push_back(T_base_to_world_calc(Rect(0, 0, 3, 3)).clone());
    }

    // --- 统计平移误差 ---
    // 计算平均位置
    mean_position.x /= board_positions_in_base.size();
    mean_position.y /= board_positions_in_base.size();
    mean_position.z /= board_positions_in_base.size();

    double total_dist_err = 0;
    double max_dist_err = 0;

    for (const auto& pos : board_positions_in_base) {
        double err = norm(pos - mean_position);
        total_dist_err += err;
        if (err > max_dist_err) max_dist_err = err;
    }
    double mean_dist_err = total_dist_err / board_positions_in_base.size();

    // --- 统计旋转误差 ---
    // 计算平均旋转矩阵 (简单平均近似，或取第一个作为基准)
    // 这里为了简单，我们计算所有解相对于标定函数输出的 "R_base_to_world_out" 的偏差
    double total_rot_err = 0;
    double max_rot_err = 0;
    
    // 如果标定函数输出了 Base->World，我们以它为真值
    Mat R_base_to_world_truth = R_base_to_world_out; 

    for (const auto& R : R_base_to_worlds) {
        // 计算 R * R_truth^T，如果是单位阵则无误差
        Mat R_diff = R * R_base_to_world_truth.t();
        Mat rvec_diff;
        Rodrigues(R_diff, rvec_diff);
        double err_deg = norm(rvec_diff) * 180.0 / CV_PI;
        
        total_rot_err += err_deg;
        if (err_deg > max_rot_err) max_rot_err = err_deg;
    }
    double mean_rot_err = total_rot_err / R_base_to_worlds.size();


    cout << "\n=== 云台标定精度评估 ===" << endl;
    cout << "逻辑：计算每一帧中标定板相对于基座的位置，统计离散程度。" << endl;
    cout << "标定板中心 (基座坐标系): [" << mean_position.x << ", " 
         << mean_position.y << ", " << mean_position.z << "] mm" << endl;
    cout << "-----------------------" << endl;
    cout << "平均平移误差 (Std Dev): " << mean_dist_err << " mm" << endl;
    cout << "最大平移误差: " << max_dist_err << " mm" << endl;
    cout << "平均旋转误差: " << mean_rot_err << " 度" << endl;
    cout << "-----------------------" << endl;
    
    if (mean_dist_err < 10.0) {
        cout << "结果评价: 正常 (误差在毫米级)" << endl;
    } else {
        cout << "结果评价: 异常 (误差较大)" << endl;
        cout << "可能原因: 1. 云台回转中心并非严格的0,0,0 (有机械偏置) \n"
             << "          2. 标定板在拍摄过程中移动了 \n"
             << "          3. 图像角点检测不准" << endl;
    }
    return 0;
}
    
cv::Point3d rotationMatrixToEulerAngles(const cv::Mat &R)
{
    // 确保矩阵是 3x3 类型，并且是 double 精度
    assert(R.rows == 3 && R.cols == 3);
    
    // 用于检测万向锁 (Gimbal Lock) 的中间量
    // sy = sqrt(r00*r00 + r10*r10)
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + 
                          R.at<double>(1, 0) * R.at<double>(1, 0));

    // 判断是否接近奇异点 (sy close to 0)
    bool singular = sy < 1e-6; 

    double x, y, z;

    if (!singular)
    {
        // 正常情况
        // Roll (绕 X 轴) = atan2(r21, r22)
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        
        // Pitch (绕 Y 轴) = atan2(-r20, sy)
        y = std::atan2(-R.at<double>(2, 0), sy);
        
        // Yaw (绕 Z 轴) = atan2(r10, r00)
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        // 万向锁情况 (Pitch = +/- 90度)
        // 此时 Yaw 和 Roll 轴重合，只能计算它们的差或和
        // 通常令 Roll = 0
        x = 0;
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
    }

    // // 将弧度转换为角度
    // x = x * 180.0 / CV_PI;
    // y = y * 180.0 / CV_PI;
    // z = z * 180.0 / CV_PI;

    // 返回 Yaw(Z), Pitch(Y), Roll(X)
    return cv::Point3d(z, y, x);
}
