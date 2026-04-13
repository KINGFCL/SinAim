/**
 * @file constrained_rect_pnp.cpp
 * @brief 带约束的矩形位姿估计器 —— 实现
 *
 * 核心算法流程（参考 OpenCV IPPE 实现并结合已知 Y 旋转角约束）：
 *
 *  数学推导：
 *    设 A→B 的内旋分解为 R_AB = R_y(γ) · R_x(β) · R_z(α)，其中 γ 已知。
 *    则 B→A 的旋转为 R_BA = R_AB^T = R_z(-α) · R_x(-β) · R_y(-γ)。
 *    令 Q = R_z(-α) · R_x(-β)（2 自由度，待求），则 R_BA = Q · R_y(-γ)。
 *
 *    角点在 A 中的位置：P_A = R_BA · P_B + t_BA = Q · [R_y(-γ) · P_B] + t_BA
 *    定义预旋转后的物体点：P'_B = R_y(-γ) · P_B
 *    则：P_A = Q · P'_B + t_BA
 *
 *    将 A 转换到 OpenCV 相机坐标系 C：
 *      P_C = R_{A→C} · P_A,  其中 R_{A→C} = [[0,-1,0],[0,0,-1],[1,0,0]]
 *    于是：P_C = (R_{A→C} · Q) · P'_B + R_{A→C} · t_BA
 *
 *    这是一个标准 PnP 问题（物体点 P'_B 到相机 C 的变换），
 *    使用 IPPE 算法对共面的 P'_B 和归一化图像点求解，
 *    得到 R_cam 和 t_cam，然后反推 Q 和 t_BA：
 *      Q = R_{C→A} · R_cam
 *      t_BA = R_{C→A} · t_cam
 *      R_BA = Q · R_y(-γ)
 */

#include "ConstrainedRectPnP.hpp"

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>  // for Rodrigues, projectPoints, SVDecomp
#include <cmath>
#include <algorithm>
#include <limits>

namespace ConstrainedRectPnP {

// ============================================================================
// 内部辅助函数（改编自 OpenCV IPPE 实现 ippe.cpp）
// ============================================================================
namespace detail {

static const double SMALL_VAL = 1e-3;

/**
 * @brief 将向量 a 旋转到 z 轴方向的旋转矩阵
 * （改编自 PoseSolver::rotateVec2ZAxis）
 */
static void rotateVec2ZAxis(const cv::Matx31d& a, cv::Matx33d& Ra)
{
    double ax = a(0), ay = a(1), az = a(2);
    double nrm = std::sqrt(ax * ax + ay * ay + az * az);
    ax /= nrm; ay /= nrm; az /= nrm;

    double c = az;
    if (std::abs(1.0 + c) < std::numeric_limits<float>::epsilon())
    {
        Ra = cv::Matx33d::zeros();
        Ra(0, 0) = 1.0; Ra(1, 1) = 1.0; Ra(2, 2) = -1.0;
    }
    else
    {
        double d = 1.0 / (1.0 + c);
        double ax2 = ax * ax, ay2 = ay * ay, axay = ax * ay;
        Ra(0, 0) = -ax2 * d + 1.0;  Ra(0, 1) = -axay * d;       Ra(0, 2) = -ax;
        Ra(1, 0) = -axay * d;       Ra(1, 1) = -ay2 * d + 1.0;  Ra(1, 2) = -ay;
        Ra(2, 0) = ax;              Ra(2, 1) = ay;               Ra(2, 2) = 1.0 - (ax2 + ay2) * d;
    }
}

/**
 * @brief 用前三个点计算将物体点旋转到 z=0 平面的旋转矩阵
 * （改编自 PoseSolver::computeObjextSpaceR3Pts）
 */
static bool computeObjSpaceR3Pts(const cv::Mat& objectPoints, cv::Matx33d& R)
{
    double p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z;
    if (objectPoints.type() == CV_64FC3)
    {
        p1x = objectPoints.at<cv::Vec3d>(0)[0]; p1y = objectPoints.at<cv::Vec3d>(0)[1]; p1z = objectPoints.at<cv::Vec3d>(0)[2];
        p2x = objectPoints.at<cv::Vec3d>(1)[0]; p2y = objectPoints.at<cv::Vec3d>(1)[1]; p2z = objectPoints.at<cv::Vec3d>(1)[2];
        p3x = objectPoints.at<cv::Vec3d>(2)[0]; p3y = objectPoints.at<cv::Vec3d>(2)[1]; p3z = objectPoints.at<cv::Vec3d>(2)[2];
    }
    else
    {
        p1x = objectPoints.at<cv::Vec3f>(0)[0]; p1y = objectPoints.at<cv::Vec3f>(0)[1]; p1z = objectPoints.at<cv::Vec3f>(0)[2];
        p2x = objectPoints.at<cv::Vec3f>(1)[0]; p2y = objectPoints.at<cv::Vec3f>(1)[1]; p2z = objectPoints.at<cv::Vec3f>(1)[2];
        p3x = objectPoints.at<cv::Vec3f>(2)[0]; p3y = objectPoints.at<cv::Vec3f>(2)[1]; p3z = objectPoints.at<cv::Vec3f>(2)[2];
    }

    double nx = (p1y - p2y) * (p1z - p3z) - (p1y - p3y) * (p1z - p2z);
    double ny = (p1x - p3x) * (p1z - p2z) - (p1x - p2x) * (p1z - p3z);
    double nz = (p1x - p2x) * (p1y - p3y) - (p1x - p3x) * (p1y - p2y);
    double nrm = std::sqrt(nx * nx + ny * ny + nz * nz);

    if (nrm > SMALL_VAL)
    {
        cv::Matx31d v(nx / nrm, ny / nrm, nz / nrm);
        rotateVec2ZAxis(v, R);
        return true;
    }
    return false;
}

/**
 * @brief 用 SVD 计算将物体点旋转到 z=0 平面的旋转矩阵
 * （改编自 PoseSolver::computeObjextSpaceRSvD）
 */
static void computeObjSpaceRSvD(const cv::Mat& UZero, cv::Matx33d& R)
{
    cv::Mat W, U, VT;
    cv::SVDecomp(UZero * UZero.t(), W, U, VT);
    cv::Mat Rt = U.t();
    if (cv::determinant(Rt) < 0)
    {
        Rt.at<double>(2, 0) = -Rt.at<double>(2, 0);
        Rt.at<double>(2, 1) = -Rt.at<double>(2, 1);
        Rt.at<double>(2, 2) = -Rt.at<double>(2, 2);
    }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R(i, j) = Rt.at<double>(i, j);
}

/**
 * @brief 将 3D 共面物体点变换到规范位置（零均值 + z=0 平面）
 * （改编自 PoseSolver::makeCanonicalObjectPoints）
 *
 * @param objectInputPoints  3D 物体点 (1xN CV_64FC3)
 * @param canonicalObjPoints 输出：2D 规范化点 (1xN CV_64FC2)
 * @param MmodelPoints2Canonical 输出：4x4 变换矩阵
 */
static void makeCanonicalObjectPoints(const cv::Mat& objectInputPoints,
                                      cv::Mat& canonicalObjPoints,
                                      cv::Mat& MmodelPoints2Canonical)
{
    int n = objectInputPoints.rows * objectInputPoints.cols;
    canonicalObjPoints.create(1, n, CV_64FC2);

    cv::Mat UZero(3, n, CV_64FC1);
    double xBar = 0, yBar = 0, zBar = 0;
    bool isOnZPlane = true;

    for (int i = 0; i < n; i++)
    {
        double x = objectInputPoints.at<cv::Vec3d>(i)[0];
        double y = objectInputPoints.at<cv::Vec3d>(i)[1];
        double z = objectInputPoints.at<cv::Vec3d>(i)[2];
        if (std::abs(z) > SMALL_VAL) isOnZPlane = false;
        xBar += x; yBar += y; zBar += z;
        UZero.at<double>(0, i) = x;
        UZero.at<double>(1, i) = y;
        UZero.at<double>(2, i) = z;
    }
    xBar /= n; yBar /= n; zBar /= n;

    for (int i = 0; i < n; i++)
    {
        UZero.at<double>(0, i) -= xBar;
        UZero.at<double>(1, i) -= yBar;
        UZero.at<double>(2, i) -= zBar;
    }

    cv::Matx44d MCenter = cv::Matx44d::eye();
    MCenter(0, 3) = -xBar; MCenter(1, 3) = -yBar; MCenter(2, 3) = -zBar;

    if (isOnZPlane)
    {
        cv::Mat(MCenter, false).copyTo(MmodelPoints2Canonical);
        for (int i = 0; i < n; i++)
        {
            canonicalObjPoints.at<cv::Vec2d>(i)[0] = UZero.at<double>(0, i);
            canonicalObjPoints.at<cv::Vec2d>(i)[1] = UZero.at<double>(1, i);
        }
    }
    else
    {
        cv::Matx33d R;
        if (!computeObjSpaceR3Pts(objectInputPoints, R))
        {
            computeObjSpaceRSvD(UZero, R);
        }

        cv::Mat UZeroAligned = cv::Mat(R) * UZero;

        for (int i = 0; i < n; i++)
        {
            canonicalObjPoints.at<cv::Vec2d>(i)[0] = UZeroAligned.at<double>(0, i);
            canonicalObjPoints.at<cv::Vec2d>(i)[1] = UZeroAligned.at<double>(1, i);
            CV_Assert(std::abs(UZeroAligned.at<double>(2, i)) <= SMALL_VAL);
        }

        cv::Matx44d MRot = cv::Matx44d::zeros();
        MRot(3, 3) = 1;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                MRot(i, j) = R(i, j);

        cv::Matx44d Mb = MRot * MCenter;
        cv::Mat(Mb, false).copyTo(MmodelPoints2Canonical);
    }
}

/**
 * @brief 各向同性数据归一化
 * （改编自 HomographyHO::normalizeDataIsotropic）
 */
static void normalizeDataIsotropic(const cv::Mat& Data, cv::Mat& DataN, cv::Mat& T, cv::Mat& Ti)
{
    int numPoints = Data.rows * Data.cols;
    int numCh = Data.channels();
    int dataType = Data.type();

    DataN.create(2, numPoints, CV_64FC1);
    T = cv::Mat::zeros(3, 3, CV_64FC1);
    Ti = cv::Mat::zeros(3, 3, CV_64FC1);

    double xm = 0, ym = 0;
    for (int i = 0; i < numPoints; i++)
    {
        if (numCh == 2)
        {
            if (dataType == CV_32FC2) { xm += Data.at<cv::Vec2f>(i)[0]; ym += Data.at<cv::Vec2f>(i)[1]; }
            else                      { xm += Data.at<cv::Vec2d>(i)[0]; ym += Data.at<cv::Vec2d>(i)[1]; }
        }
        else
        {
            if (dataType == CV_32FC3) { xm += Data.at<cv::Vec3f>(i)[0]; ym += Data.at<cv::Vec3f>(i)[1]; }
            else                      { xm += Data.at<cv::Vec3d>(i)[0]; ym += Data.at<cv::Vec3d>(i)[1]; }
        }
    }
    xm /= numPoints; ym /= numPoints;

    double kappa = 0;
    for (int i = 0; i < numPoints; i++)
    {
        double xh, yh;
        if (numCh == 2)
        {
            if (dataType == CV_32FC2) { xh = Data.at<cv::Vec2f>(i)[0] - xm; yh = Data.at<cv::Vec2f>(i)[1] - ym; }
            else                      { xh = Data.at<cv::Vec2d>(i)[0] - xm; yh = Data.at<cv::Vec2d>(i)[1] - ym; }
        }
        else
        {
            if (dataType == CV_32FC3) { xh = Data.at<cv::Vec3f>(i)[0] - xm; yh = Data.at<cv::Vec3f>(i)[1] - ym; }
            else                      { xh = Data.at<cv::Vec3d>(i)[0] - xm; yh = Data.at<cv::Vec3d>(i)[1] - ym; }
        }
        DataN.at<double>(0, i) = xh;
        DataN.at<double>(1, i) = yh;
        kappa += xh * xh + yh * yh;
    }

    double beta = std::sqrt(2.0 * numPoints / kappa);
    DataN *= beta;

    T.at<double>(0, 0) = 1.0 / beta; T.at<double>(1, 1) = 1.0 / beta;
    T.at<double>(0, 2) = xm;         T.at<double>(1, 2) = ym;
    T.at<double>(2, 2) = 1.0;

    Ti.at<double>(0, 0) = beta;        Ti.at<double>(1, 1) = beta;
    Ti.at<double>(0, 2) = -beta * xm;  Ti.at<double>(1, 2) = -beta * ym;
    Ti.at<double>(2, 2) = 1.0;
}

/**
 * @brief Harker-O'Leary 单应性估计
 * （改编自 HomographyHO::homographyHO）
 */
static void homographyHO(const cv::Mat& srcPoints, const cv::Mat& targPoints, cv::Matx33d& H)
{
    cv::Mat DataA, DataB, TA, TAi, TB, TBi;
    normalizeDataIsotropic(srcPoints, DataA, TA, TAi);
    normalizeDataIsotropic(targPoints, DataB, TB, TBi);

    int n = DataA.cols;
    CV_Assert(n == DataB.cols);

    cv::Mat C1(1, n, CV_64FC1), C2(1, n, CV_64FC1);
    cv::Mat C3(1, n, CV_64FC1), C4(1, n, CV_64FC1);
    double mC1 = 0, mC2 = 0, mC3 = 0, mC4 = 0;

    for (int i = 0; i < n; i++)
    {
        C1.at<double>(0, i) = -DataB.at<double>(0, i) * DataA.at<double>(0, i);
        C2.at<double>(0, i) = -DataB.at<double>(0, i) * DataA.at<double>(1, i);
        C3.at<double>(0, i) = -DataB.at<double>(1, i) * DataA.at<double>(0, i);
        C4.at<double>(0, i) = -DataB.at<double>(1, i) * DataA.at<double>(1, i);
        mC1 += C1.at<double>(0, i); mC2 += C2.at<double>(0, i);
        mC3 += C3.at<double>(0, i); mC4 += C4.at<double>(0, i);
    }
    mC1 /= n; mC2 /= n; mC3 /= n; mC4 /= n;

    cv::Mat Mx(n, 3, CV_64FC1), My(n, 3, CV_64FC1);
    for (int i = 0; i < n; i++)
    {
        Mx.at<double>(i, 0) = C1.at<double>(0, i) - mC1;
        Mx.at<double>(i, 1) = C2.at<double>(0, i) - mC2;
        Mx.at<double>(i, 2) = -DataB.at<double>(0, i);
        My.at<double>(i, 0) = C3.at<double>(0, i) - mC3;
        My.at<double>(i, 1) = C4.at<double>(0, i) - mC4;
        My.at<double>(i, 2) = -DataB.at<double>(1, i);
    }

    cv::Mat DataAT, DataADataAT;
    cv::transpose(DataA, DataAT);
    DataADataAT = DataA * DataAT;
    double dt = DataADataAT.at<double>(0, 0) * DataADataAT.at<double>(1, 1)
              - DataADataAT.at<double>(0, 1) * DataADataAT.at<double>(1, 0);

    cv::Mat DataADataATi(2, 2, CV_64FC1);
    DataADataATi.at<double>(0, 0) =  DataADataAT.at<double>(1, 1) / dt;
    DataADataATi.at<double>(0, 1) = -DataADataAT.at<double>(0, 1) / dt;
    DataADataATi.at<double>(1, 0) = -DataADataAT.at<double>(1, 0) / dt;
    DataADataATi.at<double>(1, 1) =  DataADataAT.at<double>(0, 0) / dt;

    cv::Mat Pp = DataADataATi * DataA;
    cv::Mat Bx = Pp * Mx;
    cv::Mat By = Pp * My;
    cv::Mat Ex = DataAT * Bx;
    cv::Mat Ey = DataAT * By;

    cv::Mat D(2 * n, 3, CV_64FC1);
    for (int i = 0; i < n; i++)
    {
        D.at<double>(i,     0) = Mx.at<double>(i, 0) - Ex.at<double>(i, 0);
        D.at<double>(i,     1) = Mx.at<double>(i, 1) - Ex.at<double>(i, 1);
        D.at<double>(i,     2) = Mx.at<double>(i, 2) - Ex.at<double>(i, 2);
        D.at<double>(i + n, 0) = My.at<double>(i, 0) - Ey.at<double>(i, 0);
        D.at<double>(i + n, 1) = My.at<double>(i, 1) - Ey.at<double>(i, 1);
        D.at<double>(i + n, 2) = My.at<double>(i, 2) - Ey.at<double>(i, 2);
    }

    cv::Mat DT, DDT;
    cv::transpose(D, DT);
    DDT = DT * D;

    cv::Mat S, U;
    cv::eigen(DDT, S, U);

    cv::Mat h789(3, 1, CV_64FC1);
    h789.at<double>(0) = U.at<double>(2, 0);
    h789.at<double>(1) = U.at<double>(2, 1);
    h789.at<double>(2) = U.at<double>(2, 2);

    cv::Mat h12 = -Bx * h789;
    cv::Mat h45 = -By * h789;
    double h3 = -(mC1 * h789.at<double>(0) + mC2 * h789.at<double>(1));
    double h6 = -(mC3 * h789.at<double>(0) + mC4 * h789.at<double>(1));

    H(0, 0) = h12.at<double>(0); H(0, 1) = h12.at<double>(1); H(0, 2) = h3;
    H(1, 0) = h45.at<double>(0); H(1, 1) = h45.at<double>(1); H(1, 2) = h6;
    H(2, 0) = h789.at<double>(0); H(2, 1) = h789.at<double>(1); H(2, 2) = h789.at<double>(2);

    cv::Matx33d TBm(TB), TAim(TAi);
    H = TBm * H * TAim;
    double h22_inv = 1.0 / H(2, 2);
    H = H * h22_inv;
}

/**
 * @brief 从单应性的 Jacobian 计算两组旋转矩阵
 * （改编自 PoseSolver::computeRotations）
 */
static void computeRotations(double j00, double j01, double j10, double j11,
                             double p, double q,
                             cv::Mat& R1, cv::Mat& R2)
{
    R1.create(3, 3, CV_64FC1);
    R2.create(3, 3, CV_64FC1);

    cv::Matx33d Rv;
    cv::Matx31d v(p, q, 1);
    rotateVec2ZAxis(v, Rv);
    Rv = Rv.t();

    double rv00 = Rv(0,0), rv01 = Rv(0,1), rv02 = Rv(0,2);
    double rv10 = Rv(1,0), rv11 = Rv(1,1), rv12 = Rv(1,2);
    double rv20 = Rv(2,0), rv21 = Rv(2,1), rv22 = Rv(2,2);

    double b00 = rv00 - p * rv20, b01 = rv01 - p * rv21;
    double b10 = rv10 - q * rv20, b11 = rv11 - q * rv21;
    double dtinv = 1.0 / (b00 * b11 - b01 * b10);

    double binv00 =  dtinv * b11, binv01 = -dtinv * b01;
    double binv10 = -dtinv * b10, binv11 =  dtinv * b00;

    double a00 = binv00 * j00 + binv01 * j10;
    double a01 = binv00 * j01 + binv01 * j11;
    double a10 = binv10 * j00 + binv11 * j10;
    double a11 = binv10 * j01 + binv11 * j11;

    double ata00 = a00 * a00 + a01 * a01;
    double ata01 = a00 * a10 + a01 * a11;
    double ata11 = a10 * a10 + a11 * a11;
    double gamma2 = 0.5 * (ata00 + ata11 + std::sqrt((ata00 - ata11) * (ata00 - ata11) + 4.0 * ata01 * ata01));
    CV_Assert(gamma2 >= 0);
    double gamma = std::sqrt(gamma2);
    CV_Assert(std::abs(gamma) > std::numeric_limits<float>::epsilon());

    double rt00 = a00 / gamma, rt01 = a01 / gamma;
    double rt10 = a10 / gamma, rt11 = a11 / gamma;

    double b0 = std::sqrt(std::max(0.0, -rt00 * rt00 - rt10 * rt10 + 1.0));
    double b1 = std::sqrt(std::max(0.0, -rt01 * rt01 - rt11 * rt11 + 1.0));
    double sp = (-rt00 * rt01 - rt10 * rt11);
    if (sp < 0) b1 = -b1;

    // 解 1
    R1.at<double>(0,0) = rt00*rv00 + rt10*rv01 + b0*rv02;
    R1.at<double>(0,1) = rt01*rv00 + rt11*rv01 + b1*rv02;
    R1.at<double>(0,2) = (b1*rt10 - b0*rt11)*rv00 + (b0*rt01 - b1*rt00)*rv01 + (rt00*rt11 - rt01*rt10)*rv02;
    R1.at<double>(1,0) = rt00*rv10 + rt10*rv11 + b0*rv12;
    R1.at<double>(1,1) = rt01*rv10 + rt11*rv11 + b1*rv12;
    R1.at<double>(1,2) = (b1*rt10 - b0*rt11)*rv10 + (b0*rt01 - b1*rt00)*rv11 + (rt00*rt11 - rt01*rt10)*rv12;
    R1.at<double>(2,0) = rt00*rv20 + rt10*rv21 + b0*rv22;
    R1.at<double>(2,1) = rt01*rv20 + rt11*rv21 + b1*rv22;
    R1.at<double>(2,2) = (b1*rt10 - b0*rt11)*rv20 + (b0*rt01 - b1*rt00)*rv21 + (rt00*rt11 - rt01*rt10)*rv22;

    // 解 2
    R2.at<double>(0,0) = rt00*rv00 + rt10*rv01 + (-b0)*rv02;
    R2.at<double>(0,1) = rt01*rv00 + rt11*rv01 + (-b1)*rv02;
    R2.at<double>(0,2) = (b0*rt11 - b1*rt10)*rv00 + (b1*rt00 - b0*rt01)*rv01 + (rt00*rt11 - rt01*rt10)*rv02;
    R2.at<double>(1,0) = rt00*rv10 + rt10*rv11 + (-b0)*rv12;
    R2.at<double>(1,1) = rt01*rv10 + rt11*rv11 + (-b1)*rv12;
    R2.at<double>(1,2) = (b0*rt11 - b1*rt10)*rv10 + (b1*rt00 - b0*rt01)*rv11 + (rt00*rt11 - rt01*rt10)*rv12;
    R2.at<double>(2,0) = rt00*rv20 + rt10*rv21 + (-b0)*rv22;
    R2.at<double>(2,1) = rt01*rv20 + rt11*rv21 + (-b1)*rv22;
    R2.at<double>(2,2) = (b0*rt11 - b1*rt10)*rv20 + (b1*rt00 - b0*rt01)*rv21 + (rt00*rt11 - rt01*rt10)*rv22;
}

/**
 * @brief 根据旋转计算对应的平移向量
 * （改编自 PoseSolver::computeTranslation）
 *
 * objectPoints: 1xN CV_64FC2 (规范化后的平面点)
 * normalizedImgPoints: 1xN CV_64FC2 (归一化图像点)
 * R: 3x3 CV_64FC1 旋转矩阵
 * t: 输出 3x1 CV_64FC1 平移向量
 */
static void computeTranslation(const cv::Mat& objectPoints, const cv::Mat& imgPoints,
                               const cv::Mat& R, cv::Mat& t)
{
    int n = objectPoints.rows * objectPoints.cols;
    t.create(3, 1, CV_64FC1);

    double ATA00 = (double)n, ATA02 = 0, ATA11 = (double)n, ATA12 = 0;
    double ATA20 = 0, ATA21 = 0, ATA22 = 0;
    double ATb0 = 0, ATb1 = 0, ATb2 = 0;

    for (int i = 0; i < n; i++)
    {
        const cv::Vec2d& objPt = objectPoints.at<cv::Vec2d>(i);
        double rx = R.at<double>(0,0)*objPt[0] + R.at<double>(0,1)*objPt[1];
        double ry = R.at<double>(1,0)*objPt[0] + R.at<double>(1,1)*objPt[1];
        double rz = R.at<double>(2,0)*objPt[0] + R.at<double>(2,1)*objPt[1];

        const cv::Vec2d& ip = imgPoints.at<cv::Vec2d>(i);
        double a2 = -ip[0], b2 = -ip[1];

        ATA02 += a2; ATA12 += b2; ATA20 += a2; ATA21 += b2;
        ATA22 += a2*a2 + b2*b2;

        double bx = -a2*rz - rx;
        double by = -b2*rz - ry;
        ATb0 += bx; ATb1 += by; ATb2 += a2*bx + b2*by;
    }

    double detAInv = 1.0 / (ATA00*ATA11*ATA22 - ATA00*ATA12*ATA21 - ATA02*ATA11*ATA20);

    double S00 = ATA11*ATA22 - ATA12*ATA21;
    double S01 = ATA02*ATA21;
    double S02 = -ATA02*ATA11;
    double S10 = ATA12*ATA20;
    double S11 = ATA00*ATA22 - ATA02*ATA20;
    double S12 = -ATA00*ATA12;
    double S20 = -ATA11*ATA20;
    double S21 = -ATA00*ATA21;
    double S22 = ATA00*ATA11;

    t.at<double>(0) = detAInv * (S00*ATb0 + S01*ATb1 + S02*ATb2);
    t.at<double>(1) = detAInv * (S10*ATb0 + S11*ATb1 + S12*ATb2);
    t.at<double>(2) = detAInv * (S20*ATb0 + S21*ATb1 + S22*ATb2);
}

/**
 * @brief 评估重投影误差（RMSE，在归一化图像平面上）
 * （改编自 PoseSolver::evalReprojError）
 */
static double evalReprojError(const cv::Mat& objectPoints3D, const cv::Mat& imagePoints,
                              const cv::Mat& R, const cv::Mat& t)
{
    // objectPoints3D: 1xN CV_64FC3
    // imagePoints: 1xN CV_64FC2
    // 在归一化平面上计算重投影误差（K = I, dist = 0）
    int n = objectPoints3D.rows * objectPoints3D.cols;
    cv::Mat rvec;
    cv::Rodrigues(R, rvec);

    cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
    cv::Mat dist;
    cv::Mat projPts;
    cv::projectPoints(objectPoints3D, rvec, t, K, dist, projPts);

    double err = 0;
    for (int i = 0; i < n; i++)
    {
        double dx, dy;
        if (projPts.depth() == CV_32F)
        {
            dx = projPts.at<cv::Vec2f>(i)[0] - imagePoints.at<cv::Vec2d>(i)[0];
            dy = projPts.at<cv::Vec2f>(i)[1] - imagePoints.at<cv::Vec2d>(i)[1];
        }
        else
        {
            dx = projPts.at<cv::Vec2d>(i)[0] - imagePoints.at<cv::Vec2d>(i)[0];
            dy = projPts.at<cv::Vec2d>(i)[1] - imagePoints.at<cv::Vec2d>(i)[1];
        }
        err += dx * dx + dy * dy;
    }
    return std::sqrt(err / (2.0 * n));
}

/**
 * @brief IPPE 核心求解（在规范位置）
 * （改编自 PoseSolver::solveCanonicalForm）
 *
 * 输出两个 4x4 位姿矩阵 Ma, Mb
 */
static void solveCanonicalForm(const cv::Mat& canonicalObjPoints, const cv::Mat& normalizedImgPoints,
                               const cv::Matx33d& H,
                               cv::Mat& Ma, cv::Mat& Mb)
{
    Ma = cv::Mat::zeros(4, 4, CV_64FC1);
    Ma.at<double>(3, 3) = 1;
    Mb = cv::Mat::zeros(4, 4, CV_64FC1);
    Mb.at<double>(3, 3) = 1;

    // 计算单应性在 (0,0) 处的 Jacobian
    double j00 = H(0,0) - H(2,0)*H(0,2);
    double j01 = H(0,1) - H(2,1)*H(0,2);
    double j10 = H(1,0) - H(2,0)*H(1,2);
    double j11 = H(1,1) - H(2,1)*H(1,2);

    // (0,0) 在图像中的映射
    double v0 = H(0, 2);
    double v1 = H(1, 2);

    // 计算两个旋转解
    cv::Mat Ra = Ma.colRange(0, 3).rowRange(0, 3);
    cv::Mat Rb = Mb.colRange(0, 3).rowRange(0, 3);
    computeRotations(j00, j01, j10, j11, v0, v1, Ra, Rb);

    // 计算对应的平移解
    cv::Mat ta = Ma.colRange(3, 4).rowRange(0, 3);
    cv::Mat tb = Mb.colRange(3, 4).rowRange(0, 3);
    computeTranslation(canonicalObjPoints, normalizedImgPoints, Ra, ta);
    computeTranslation(canonicalObjPoints, normalizedImgPoints, Rb, tb);
}

/**
 * @brief IPPE 完整求解流程（处理一般共面 3D 点）
 *
 * @param objPoints3D  3D 共面物体点 (1xN CV_64FC3)
 * @param normImgPts   归一化图像点 (1xN CV_64FC2)
 * @param R1, t1, err1 第一个解
 * @param R2, t2, err2 第二个解
 */
static void solveIPPE(const cv::Mat& objPoints3D, const cv::Mat& normImgPts,
                      cv::Mat& R1, cv::Mat& t1, double& err1,
                      cv::Mat& R2, cv::Mat& t2, double& err2)
{
    // 1. 变换到规范位置
    cv::Mat canonicalObjPoints, MmodelPoints2Canonical;
    makeCanonicalObjectPoints(objPoints3D, canonicalObjPoints, MmodelPoints2Canonical);

    // 2. 计算单应性：从规范物体点到归一化图像点
    cv::Matx33d H;
    homographyHO(canonicalObjPoints, normImgPts, H);

    // 3. 在规范位置求解
    cv::Mat MaCanon, MbCanon;
    solveCanonicalForm(canonicalObjPoints, normImgPts, H, MaCanon, MbCanon);

    // 4. 变换回原始物体坐标系
    cv::Mat Ma = MaCanon * MmodelPoints2Canonical;
    cv::Mat Mb = MbCanon * MmodelPoints2Canonical;

    // 5. 提取 R, t 并按重投影误差排序
    cv::Mat Ra = Ma.colRange(0, 3).rowRange(0, 3);
    cv::Mat ta = Ma.colRange(3, 4).rowRange(0, 3);
    cv::Mat Rb = Mb.colRange(0, 3).rowRange(0, 3);
    cv::Mat tb = Mb.colRange(3, 4).rowRange(0, 3);

    double erra = evalReprojError(objPoints3D, normImgPts, Ra, ta);
    double errb = evalReprojError(objPoints3D, normImgPts, Rb, tb);

    if (erra <= errb)
    {
        Ra.copyTo(R1); ta.copyTo(t1); err1 = erra;
        Rb.copyTo(R2); tb.copyTo(t2); err2 = errb;
    }
    else
    {
        Rb.copyTo(R1); tb.copyTo(t1); err1 = errb;
        Ra.copyTo(R2); ta.copyTo(t2); err2 = erra;
    }
}

} // namespace detail

// ============================================================================
// 公开接口
// ============================================================================

void decomposeIntrinsicZXY(const cv::Matx33d& R_AB,
                           double& alpha, double& beta, double& gamma)
{
    /*
     * R_AB = R_y(gamma) * R_x(beta) * R_z(alpha)
     *
     * 展开后矩阵为：
     *   [ cγ·cα + sγ·sα·sβ,  -cγ·sα + sγ·cα·sβ,  sγ·cβ ]
     *   [       sα·cβ,              cα·cβ,            -sβ   ]
     *   [-sγ·cα + cγ·sα·sβ,  sγ·sα + cγ·cα·sβ,   cγ·cβ ]
     *
     * 其中 sα=sin(α), cα=cos(α), 以此类推。
     *
     * 提取公式：
     *   β = asin(-R(1,2))
     *   若 cos(β) ≠ 0：
     *     α = atan2(R(1,0), R(1,1))
     *     γ = atan2(R(0,2), R(2,2))
     */
    beta = std::asin(-R_AB(1, 2));
    double cb = std::cos(beta);

    if (std::abs(cb) > 1e-10)
    {
        alpha = std::atan2(R_AB(1, 0), R_AB(1, 1));
        gamma = std::atan2(R_AB(0, 2), R_AB(2, 2));
    }
    else
    {
        // 万向节锁：β ≈ ±90°，此时 α 和 γ 耦合
        alpha = 0;
        gamma = std::atan2(-R_AB(2, 0), R_AB(0, 0));
    }
}

int solve(const std::vector<cv::Vec3d>& dirVecsInA,
          const std::vector<cv::Vec3d>& cornerPtsInB,
          double knownYAngle,
          std::vector<PoseResult>& results)
{
    CV_Assert(dirVecsInA.size() == 4);
    CV_Assert(cornerPtsInB.size() == 4);

    results.clear();

    // ================================================================
    // 第一步：构造 R_y(-γ)，对物体点做预旋转
    // ================================================================
    // R_y(θ) = [[cos θ, 0, sin θ], [0, 1, 0], [-sin θ, 0, cos θ]]
    // R_y(-γ) = [[cos γ, 0, -sin γ], [0, 1, 0], [sin γ, 0, cos γ]]
    double cg = std::cos(knownYAngle), sg = std::sin(knownYAngle);
    cv::Matx33d Ry_neg_gamma(
         cg, 0, -sg,
          0, 1,   0,
         sg, 0,  cg
    );

    // P'_B = R_y(-γ) * P_B
    // 原始 P_B = (0, y_i, z_i)（矩形在 YZ 平面上）
    // P'_B = (-sin(γ)*z_i, y_i, cos(γ)*z_i) —— 仍然共面
    cv::Mat objPtsPreRotated(1, 4, CV_64FC3);
    for (int i = 0; i < 4; i++)
    {
        cv::Vec3d p = cornerPtsInB[i];
        cv::Vec3d pp = Ry_neg_gamma * p;
        objPtsPreRotated.at<cv::Vec3d>(i) = pp;
    }

    // ================================================================
    // 第二步：将 A 中的方向向量转换为 OpenCV 相机坐标系的归一化图像点
    // ================================================================
    // 坐标系映射：A_x→CV_z, A_y→-CV_x, A_z→-CV_y
    // 归一化图像坐标：u = CV_x/CV_z = -A_y/A_x,  v = CV_y/CV_z = -A_z/A_x
    cv::Mat normImgPts(1, 4, CV_64FC2);
    for (int i = 0; i < 4; i++)
    {
        const cv::Vec3d& d = dirVecsInA[i];
        CV_Assert(d[0] > 0);  // 方向向量的 x 分量必须为正（目标在 A 的正 x 方向上）
        double u = -d[1] / d[0];
        double v = -d[2] / d[0];
        normImgPts.at<cv::Vec2d>(i) = cv::Vec2d(u, v);
    }

    // ================================================================
    // 第三步：使用 IPPE 算法求解
    // ================================================================
    // IPPE 输入：
    //   - 物体点：P'_B（3D 共面点，在 OpenCV 物体坐标系中）
    //   - 图像点：归一化像素坐标（相当于 K=I, dist=0 的 undistortPoints 输出）
    //
    // IPPE 输出：
    //   - R_cam: 物体坐标系 → 相机坐标系的旋转
    //   - t_cam: 物体坐标系 → 相机坐标系的平移
    //
    // 关系推导：
    //   P_C = R_cam * P'_B + t_cam
    //   又 P_C = R_{A→C} * P_A = R_{A→C} * (Q * P'_B + t_BA)
    //         = (R_{A→C} * Q) * P'_B + R_{A→C} * t_BA
    //   所以：R_cam = R_{A→C} * Q,  t_cam = R_{A→C} * t_BA
    //   反推：Q = R_{C→A} * R_cam,  t_BA = R_{C→A} * t_cam
    //   最终：R_BA = Q * R_y(-γ)

    cv::Mat R_sol1, t_sol1, R_sol2, t_sol2;
    double err1, err2;

    try
    {
        detail::solveIPPE(objPtsPreRotated, normImgPts,
                          R_sol1, t_sol1, err1,
                          R_sol2, t_sol2, err2);
    }
    catch (const cv::Exception&)
    {
        return 0;  // 求解失败（数据退化等）
    }

    // ================================================================
    // 第四步：将 IPPE 结果转换回 A-B 坐标系关系
    // ================================================================
    // R_{A→C} = [[0,-1,0],[0,0,-1],[1,0,0]]
    // R_{C→A} = R_{A→C}^T = [[0,0,1],[-1,0,0],[0,-1,0]]
    cv::Matx33d R_C2A(
         0,  0, 1,
        -1,  0, 0,
         0, -1, 0
    );

    cv::Mat R_cams[2]  = { R_sol1, R_sol2 };
    cv::Mat t_cams[2]  = { t_sol1, t_sol2 };
    double  errs[2]    = { err1,   err2   };

    for (int k = 0; k < 2; k++)
    {
        // 提取 3x3 旋转和 3x1 平移
        cv::Matx33d R_cam;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R_cam(i, j) = R_cams[k].at<double>(i, j);

        cv::Vec3d t_cam(t_cams[k].at<double>(0),
                        t_cams[k].at<double>(1),
                        t_cams[k].at<double>(2));

        // Q = R_{C→A} * R_cam
        cv::Matx33d Q = R_C2A * R_cam;

        // R_BA = Q * R_y(-γ)
        cv::Matx33d R_BA = Q * Ry_neg_gamma;

        // t_BA = R_{C→A} * t_cam
        cv::Vec3d t_BA = R_C2A * t_cam;

        // ============================================================
        // 深度检查：B 的原点在 A 的 x 正方向（即 t_BA[0] > 0）
        // 与 OpenCV solvePnP 中"物体在相机前方"约束等价
        // ============================================================
        // if (t_BA[0] < 0)
        // {
        //     // 翻转（IPPE 的两个解通常都满足正深度，
        //     // 但以防万一进行处理）
        //     t_BA = -t_BA;
        //     // 注意：如果需要翻转 t，通常意味着该解不太物理，
        //     // 但仍然作为候选返回
        // }

        // 分解内旋 Z-X-Y 欧拉角
        // R_AB = R_BA^T
        cv::Matx33d R_AB = R_BA.t();
        double alpha_z, beta_x, gamma_y;
        decomposeIntrinsicZXY(R_AB, alpha_z, beta_x, gamma_y);

        results.emplace_back(R_BA, t_BA, errs[k], alpha_z, beta_x, gamma_y);
    }

    // 按重投影误差从小到大排序
    if (results.size() == 2 && results[0].reprojError > results[1].reprojError)
    {
        std::swap(results[0], results[1]);
    }

    return static_cast<int>(results.size());
}

} // namespace ConstrainedRectPnP

