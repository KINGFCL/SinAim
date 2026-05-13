#include "OPEKFKalman.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core/cvdef.h>

OPEKFKalman::OPEKFKalman(const Eigen::Matrix3d& RCamera2Grip)
    : RCamera2Grip(RCamera2Grip)
{
    this->Init();
}

void OPEKFKalman::Init()
{
    this->CovState = (StateVector() <<
        100.0, 100.0, 100.0,
        0.01, 5.0,
        0.0, 0.0, 0.0
    ).finished().asDiagonal();
}

OPEKFKalman::StateVector OPEKFKalman::Predict(const StateVector& state, double dt)
{
    StateVector x_curr = state;
    x_curr(3, 0) = std::remainder(x_curr(3, 0) + x_curr(4, 0) * dt, CV_PI * 2.0);

    StateMatrix F = StateMatrix::Identity();
    F(3, 4) = dt;

    StateMatrix Q = StateMatrix::Zero();
    const double a = dt * dt * dt * dt * 0.25;
    const double b = dt * dt * dt * 0.5;
    const double c = dt * dt;

    Q(3, 3) = a * this->Var_alpha;
    Q(4, 4) = c * this->Var_alpha;
    Q(3, 4) = Q(4, 3) = b * this->Var_alpha;
    Q(2, 2) = c * this->Var_a_z;
    Q(5, 5) = this->Var_radius * dt;
    Q(6, 6) = this->Var_height_diff * dt;
    Q(7, 7) = this->Var_height_diff * dt;

    this->CovState = F * this->CovState * F.transpose() + Q;
    return x_curr;
}

OPEKFKalman::StateVector OPEKFKalman::Correct(
    const StateVector& state,
    const Eigen::Matrix<double, 4, 1>& view,
    const Eigen::Vector3d& scs,
    const Eigen::Quaterniond& gripper_to_world,
    double delta_angle,
    size_t armor_id,
    double dt)
{
    StateVector x_curr = this->Predict(state, dt);

    Eigen::Matrix<double, 4, 4> cov_view = Eigen::Matrix<double, 4, 4>::Zero();
    Eigen::Matrix3d cov_camera = (Eigen::Vector3d{
        (std::log(std::abs(delta_angle) + 1.0) + 1.0) * this->Var_r,
        0.009,
        0.009
    }).asDiagonal();

    Eigen::Matrix3d jacobian_s2c = this->GetJacobianSphericalToCartesian(scs);
    Eigen::Matrix3d cov_camera_ccs = jacobian_s2c * cov_camera * jacobian_s2c.transpose();
    Eigen::Matrix3d r_cam_to_world = gripper_to_world.toRotationMatrix() * this->RCamera2Grip;
    cov_view.block<3, 3>(0, 0) = r_cam_to_world * cov_camera_ccs * r_cam_to_world.transpose();

    const double yaw_var_standard =
        (std::log(std::max((view.block<3, 1>(0, 0).norm() / 100.0) - 3.0, 0.0) + 1.0) + 1.0) *
        this->Var_yaw;
    cov_view(3, 3) = std::exp(std::abs(delta_angle) - (CV_PI / 4.0)) * yaw_var_standard;

    Eigen::Matrix<double, 4, 1> view_curr = this->StateToView(x_curr, armor_id);
    Eigen::Matrix<double, 4, 1> innovation = view - view_curr;
    innovation(3, 0) = std::remainder(innovation(3, 0), CV_PI * 2.0);

    const double s_yaw = this->GetAngleRobustScale(std::abs(innovation(3, 0)));
    Eigen::Matrix<double, 4, 4> robust = (Eigen::Matrix<double, 4, 1>{1.0, 1.0, 1.0, s_yaw}).asDiagonal();
    cov_view = robust * cov_view * robust.transpose();

    Eigen::Matrix<double, 4, 8> H = this->GetStateToViewJacobian(x_curr, armor_id);
    Eigen::Matrix<double, 8, 4> K =
        this->CovState * H.transpose() * (H * this->CovState * H.transpose() + cov_view).inverse();

    StateVector x_next = x_curr + K * innovation;
    x_next(3, 0) = std::remainder(x_next(3, 0), CV_PI * 2.0);

    StateMatrix I_KH = StateMatrix::Identity() - K * H;
    this->CovState = I_KH * this->CovState * I_KH.transpose() + K * cov_view * K.transpose();

    return x_next;
}

Eigen::Matrix<double, 4, 1> OPEKFKalman::StateToView(const StateVector& state, size_t armor_id) const
{
    const double yaw_i = this->ArmorYaw(state(3, 0), armor_id);

    Eigen::Matrix<double, 4, 1> view;
    view(0, 0) = state(0, 0) + state(5, 0) * std::cos(yaw_i);
    view(1, 0) = state(1, 0) + state(5, 0) * std::sin(yaw_i);
    view(2, 0) = this->ArmorHeight(state(2, 0), state(6, 0), state(7, 0), armor_id);
    view(3, 0) = yaw_i;
    return view;
}

Eigen::Matrix<double, 4, 8> OPEKFKalman::GetStateToViewJacobian(
    const StateVector& state,
    size_t armor_id) const
{
    Eigen::Matrix<double, 4, 8> H = Eigen::Matrix<double, 4, 8>::Zero();

    const double yaw_i = this->ArmorYaw(state(3, 0), armor_id);
    const double cos_yaw = std::cos(yaw_i);
    const double sin_yaw = std::sin(yaw_i);

    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 2) = 1.0;

    H(0, 3) = -state(5, 0) * sin_yaw;
    H(1, 3) = state(5, 0) * cos_yaw;
    H(3, 3) = 1.0;

    H(0, 5) = cos_yaw;
    H(1, 5) = sin_yaw;
    if (armor_id == 1) H(2, 6) = 1.0;
    if (armor_id == 2) H(2, 7) = 1.0;

    return H;
}

Eigen::Matrix3d OPEKFKalman::GetJacobianSphericalToCartesian(const Eigen::Vector3d& scs) const
{
    const double radius = scs(0);
    const double theta = scs(1);
    const double phi = scs(2);

    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double sp = std::sin(phi);
    const double cp = std::cos(phi);

    Eigen::Matrix3d J;
    J(0, 0) = st * cp;
    J(0, 1) = radius * ct * cp;
    J(0, 2) = -radius * st * sp;
    J(1, 0) = st * sp;
    J(1, 1) = radius * ct * sp;
    J(1, 2) = radius * st * cp;
    J(2, 0) = ct;
    J(2, 1) = -radius * st;
    J(2, 2) = 0.0;
    return J;
}

double OPEKFKalman::GetAngleRobustScale(double angle_error) const
{
    const double p2 = angle_error * angle_error;
    const double p4 = p2 * p2;
    const double p6 = p4 * p2;
    const double t2 = this->Threshold_Angle * this->Threshold_Angle;
    const double t4 = t2 * t2;
    const double t6 = t4 * t2;
    const double scale = 1.0 + (this->Max_Robust_Scale - 1.0) * (p6 / (p6 + t6));
    return std::sqrt(scale);
}

double OPEKFKalman::ArmorYaw(double yaw0, size_t armor_id) const
{
    return std::remainder(yaw0 + static_cast<double>(armor_id) * ARMOR_ANGLE_STEP, CV_PI * 2.0);
}

double OPEKFKalman::ArmorHeight(double z0, double dh1, double dh2, size_t armor_id) const
{
    if (armor_id == 1) return z0 + dh1;
    if (armor_id == 2) return z0 + dh2;
    return z0;
}
