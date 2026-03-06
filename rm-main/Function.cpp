#include "Function.hpp"
#include <chrono>
#include <ratio>
#include <sys/types.h>

void rm::IMUAndImageMatchFunction(io::HikCamera &Hik, io::RTSerial<Packet> &ser, FastQueue<FrameData> &Frames)
{
    while (true) {

        // 读取相机数据
        io::HikCamera::ImageData HikData;

        Hik.read(HikData);
        if(HikData.image.empty()) continue;

        // 读取串口数据
        std::chrono::steady_clock::time_point time ;
        Packet IMU;
        while( true )
        {
            bool ret = ser.readPacket(IMU, time);
            if( !ret ) break;

            const auto& t = ((double)(HikData.time - time).count()) * 1e-6;

            //配对超时

            //串口数据比相机数据早8ms以上
            if( t > 8 ) continue;

            //串口数据比相机数据早5ms以下
            if( t < 5 ) break;

            //配对成功
            cv::Quatd quat( IMU.q3, IMU.q0, IMU.q1, IMU.q2 );
            FrameData frame(HikData.image, quat, HikData.time);

            Frames.push(frame);
            break;
        }

    }
}

void rm::SendMessageToRobot(io::RTSerial<Packet> &ser, float pitch, float yaw, bool fire)
{
    ShootPosi posimsg{pitch, yaw};
    ShootFire firemsg{fire};

    ser.writeBytes(&posimsg,sizeof(posimsg));
    ser.writeBytes(&firemsg,sizeof(firemsg));
}

Eigen::Matrix<double, 3, 1> rm::ChooseBestAimArmor(
    const Eigen::Matrix<double, 4, 4>& aims,
    const Eigen::Matrix<double, 4, 1>& Speed, // 修正为 4x1 (vx, vy, vz, w)
    const Eigen::Matrix<double, 3, 1>& Gun    // 当前枪管朝向（需为单位向量）
) 
{
    int best_id = -1;
    double min_cost = std::numeric_limits<double>::max();
    
    // 提取车体自旋角速度 w
    const double& w = Speed(3, 0); 
    
    for (int i = 0; i < 4; i++) 
    {
        // 提取预测的第 i 个装甲板的位置和偏航角
        Eigen::Matrix<double, 3, 1> P_i = aims.block<3, 1>(0, i);
        double theta_i = aims(3, i);
        
        double distance = P_i.norm();
        if (distance < 1e-3) continue; // 规避奇异点(除0)
        
        // 1. 视线向量 L_i (从枪管指向装甲板的单位向量)
        Eigen::Matrix<double, 3, 1> L_i = P_i / distance; 
        
        // 2. 装甲板法向量 N_i (垂直于装甲板向外指的单位向量)
        Eigen::Matrix<double, 3, 1> N_i {std::cos(theta_i), std::sin(theta_i), 0.0};
        
        // -------------------------------------------------------------
        // A. 迎角投影 (Facing Projection)
        // 计算视线与法向量的点乘。由于 L_i 指向目标，N_i 指向我们，二者反向时最优
        // face_proj 范围：1.0 (完美正对) 到 -1.0 (完全背对)
        // -------------------------------------------------------------
        double face_proj = -N_i.dot(L_i);
        
        // 【核心硬截断】：如果装甲板背对我们，或者侧得太厉害（夹角 > 80度左右），直接放弃
        // 0.17 约等于 cos(80度)，侧转太厉害的装甲板不仅打不中，弹丸也容易跳弹
        if (face_proj < 0.17) {
            continue; 
        }
        
        // -------------------------------------------------------------
        // B. 枪管追踪投影 (Tracking Projection)
        // 计算当前枪管 Gun 和目标视线 L_i 的对齐程度
        // tracking_proj 范围：1.0 (无需转动云台) 到 -1.0 (需要向后转180度)
        // -------------------------------------------------------------
        double tracking_proj = Gun.dot(L_i);
        
        // -------------------------------------------------------------
        // C. 旋转趋势奖励 (Spin Margin) - 应对小陀螺的顶级策略
        // 利用叉积计算当前装甲板正在"转向我们"还是"背离我们"
        // -------------------------------------------------------------
        double cross_z = N_i(0) * L_i(1) - N_i(1) * L_i(0); // N_i x L_i 的 Z 轴分量
        double spin_trend = w * cross_z; 
        // 若 spin_trend > 0，说明装甲板正顺着旋转趋势转入我们的最佳视线
        // 若 spin_trend < 0，说明装甲板即将转出视线，容错率极低
        
        // -------------------------------------------------------------
        // 综合代价函数 (Cost Function)
        // 权重设计：Cost 越小越好。将 proj 转换为 (1 - proj) 使其成为惩罚项。
        // 可根据实车联调表现微调权重。
        // -------------------------------------------------------------
        const double w_face = 1.0;  // 权重1：对装甲板正度的重视程度
        const double w_track = 1.5; // 权重2：对云台响应速度的重视程度（通常云台物理响应最慢，给高权重）
        const double w_spin = 0.5;  // 权重3：对旋转提前量的奖励权重
        
        double cost = w_face * (1.0 - face_proj) 
                    + w_track * (1.0 - tracking_proj) 
                    - w_spin * spin_trend; // 奖励正向转入的装甲板，惩罚背离的
        
        if (cost < min_cost) {
            min_cost = cost;
            best_id = i;
        }
    }
    
    // -------------------------------------------------------------
    // 异常安全保护 (Fallback)
    // -------------------------------------------------------------
    // 如果预测由于某些极端原因（如剧烈畸变）导致所有装甲板都被硬截断剔除了，
    // 则强制选择一个迎角最大的装甲板，保证即使在最差情况下云台也不会发呆。
    if (best_id == -1) {
        best_id = 0;
        double max_face = -2.0;
        for (int i = 0; i < 4; i++) {
            Eigen::Matrix<double, 3, 1> L_i = aims.block<3, 1>(0, i).normalized();
            Eigen::Matrix<double, 3, 1> N_i {std::cos(aims(3,i)), std::sin(aims(3,i)), 0.0};
            double face = -N_i.dot(L_i);
            if (face > max_face) {
                max_face = face;
                best_id = i;
            }
        }
    }
    
    // 返回被选中的最优装甲板的 3D 绝对坐标
    return aims.block<3, 1>(0, best_id);
}

double rm::SolveDt(const std::chrono::steady_clock::time_point& start, const std::chrono::steady_clock::time_point& end, double pic) {
    // 1. 安全检查：避免除以零
    if (std::abs(pic) < 1e-9) {
        return 0.0; 
    }

    // 2. 获取时间差，并将单位转换为秒 (s)
    // std::chrono::duration<double> 默认就等同于 std::chrono::duration<double, std::ratio<1>>，也就是秒。
    std::chrono::duration<double> diff = end - start;
    double dt = diff.count();

    // 3. 寻找最接近的倍数 n
    double n = std::round(dt / pic);

    // 4. 保证 n 为非零整数
    if (n == 0.0) {
        n = (dt >= 0) ? 1.0 : -1.0;
    }

    // 5. 返回最接近的 n * pic (单位：秒)
    return n * pic;
}
