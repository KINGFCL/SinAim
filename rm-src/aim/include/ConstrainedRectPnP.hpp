/**
 * @file constrained_rect_pnp.hpp
 * @brief 带约束的矩形位姿估计器
 *
 * 问题描述：
 *   已知世界坐标系 A，矩形局部坐标系 B（以矩形几何中心为原点，Y沿宽度向左，Z沿高度向上，X为法向外）。
 *   已知矩形四个角点在 A 中的方向向量（模长未知），以及角点在 B 中的坐标。
 *   A→B 的内旋顺序为 Z→X→Y，其中 Y 旋转角已知，Z 和 X 旋转角未知。
 *   求 B→A 的旋转矩阵及矩形中心在 A 中的方向向量。
 *
 * 算法：参考 OpenCV 官方 IPPE (Infinitesimal Plane-based Pose Estimation) 实现，
 *       利用已知 Y 旋转角对物体点进行预旋转以降低问题自由度，
 *       然后通过单应性分解得到两组解（对应 Z 旋转方向的正/反两种可能）。
 *
 * 坐标系映射关系（A 与 OpenCV 相机坐标系）：
 *   A_x+ ↔ CV_z+  (A 的 x 正方向 = 相机光轴正方向)
 *   A_y+ ↔ CV_x-  (A 的 y 正方向 = 相机 x 轴负方向)
 *   A_z+ ↔ CV_y-  (A 的 z 正方向 = 相机 y 轴负方向)
 */

#ifndef CONSTRAINED_RECT_PNP_HPP
#define CONSTRAINED_RECT_PNP_HPP

#include <opencv2/core.hpp>
#include <vector>

namespace ConstrainedRectPnP {

/**
 * @brief 单个位姿解的结果
 */
struct PoseResult {
    cv::Matx33d R_B2A;         ///< B→A 的旋转矩阵
    cv::Vec3d   centerDirInA;  ///< 矩形中心在 A 中的单位方向向量
    double      reprojError;   ///< 重投影误差（归一化图像平面上的 RMSE）

    // 以下是从旋转矩阵中分解出的内旋 Z-X-Y 欧拉角（供参考/筛选用）
    double alpha_z;  ///< 第一步内旋绕 Z 轴角度 (rad)
    double beta_x;   ///< 第二步内旋绕 X 轴角度 (rad)
    double gamma_y;  ///< 第三步内旋绕 Y 轴角度 (rad)（应接近输入的 knownYAngle）
};

/**
 * @brief 求解带约束的矩形位姿
 *
 * @param dirVecsInA    矩形 4 个角点在坐标系 A 中的方向向量（无需单位化，但 x 分量必须 > 0）。
 *                      顺序须与 cornerPtsInB 一一对应。
 * @param cornerPtsInB  矩形 4 个角点在坐标系 B 中的坐标。
 *                      由于角点在矩形平面上，x 分量应为 0。
 * @param knownYAngle   内旋分解 Z→X→Y 中已知的 Y 旋转角 (弧度)。
 * @param results       输出：恰好 2 个解，按重投影误差从小到大排序。
 *                      两个解分别对应绕 Z 轴旋转方向为逆时针/顺时针的两种情况。
 * @return 找到的解的数量（正常为 2；如果数据退化可能为 0）
 */
int solve(const std::vector<cv::Vec3d>& dirVecsInA,
          const std::vector<cv::Vec3d>& cornerPtsInB,
          double knownYAngle,
          std::vector<PoseResult>& results);

/**
 * @brief 从 R_AB（A→B 的旋转矩阵）中提取内旋 Z-X-Y 欧拉角
 *
 * R_AB = R_y(gamma) * R_x(beta) * R_z(alpha)
 *
 * @param R_AB    A→B 的旋转矩阵
 * @param alpha   输出：绕 Z 轴旋转角 (rad)
 * @param beta    输出：绕 X 轴旋转角 (rad)
 * @param gamma   输出：绕 Y 轴旋转角 (rad)
 */
void decomposeIntrinsicZXY(const cv::Matx33d& R_AB,
                           double& alpha, double& beta, double& gamma);

} // namespace ConstrainedRectPnP

#endif // CONSTRAINED_RECT_PNP_HPP

