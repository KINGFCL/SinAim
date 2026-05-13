#include "OutPust.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/core/cvdef.h>

OutPust::OutPust(const OutPustConfig& config)
    : ekfkalman(Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
          config.gripper_to_world_matrix.data()))
{
}

void OutPust::Clear()
{
    this->is_init = false;
    this->has_preinit_armor = false;
    this->ekfkalman.Init();
    this->View = {ArmorView::Invisual, ArmorView::Invisual, ArmorView::Invisual};
}

void OutPust::ResetState(const Eigen::Vector3d& center,
                         double yaw,
                         double w,
                         double r,
                         double d_h1,
                         double d_h2)
{
    this->is_init = true;
    this->has_preinit_armor = false;
    this->ekfkalman.Init();

    this->center = center;
    this->yaw = std::remainder(yaw, CV_PI * 2.0);
    this->w = w;
    this->r = r;
    this->d_h1 = d_h1;
    this->d_h2 = d_h2;
    this->Speed << 0.0, 0.0, 0.0, this->w;
    this->UpdateArmorState();
}

void OutPust::Update(const std::vector<ArmorPosi>& armors, const Eigen::Quaterniond& gripper_to_world, double dt)
{
    if (armors.empty()) {
        this->Update(dt);
        return;
    }

    const ArmorPosi& armor = armors.front();
    const size_t side = this->BestSide(armor);

    if (!this->is_init) {
        this->TryInitByJump(armor, side);
        return;
    }

    this->OneArmor(armor, gripper_to_world, dt);
}

void OutPust::Update(double dt)
{
    if (!this->is_init) return;

    StateVector ans = this->ekfkalman.Predict(this->MakeState(), dt);
    this->ApplyState(ans);
}

Eigen::Matrix<double, 4, 3> OutPust::Predict(double dt) const
{
    Eigen::Matrix<double, 4, 3> ans;

    const double yaw0 = std::remainder(this->yaw + this->w * dt, CV_PI * 2.0);
    for (size_t id = 0; id < ARMOR_NUM; ++id) {
        const double yaw_i = this->ArmorYaw(yaw0, id);
        ans(0, id) = this->center(0) + this->r * std::cos(yaw_i);
        ans(1, id) = this->center(1) + this->r * std::sin(yaw_i);
        ans(2, id) = this->ArmorHeight(this->center(2), id);
        ans(3, id) = yaw_i;
    }

    return ans;
}

Eigen::Matrix<double, 4, 3> OutPust::Predict(double dt, Eigen::Vector3d& Center) const
{
    Center = this->center;
    return this->Predict(dt);
}

void OutPust::TryInitByJump(const ArmorPosi& armor, size_t side)
{
    const Eigen::Vector3d armor_center = armor.center.col(side);
    const Eigen::Vector3d armor_scs = armor.SCS.col(side);
    const double armor_yaw = armor.yaw[side];
    const double armor_theta = armor.theta[side];

    if (!this->has_preinit_armor) {
        this->SavePreinit(armor_center, armor_scs, armor.photocenter, armor_yaw, armor_theta);
        return;
    }

    Eigen::Vector3d old_ray = this->preinit_center - this->preinit_photocenter;
    Eigen::Vector3d new_ray = armor_center - armor.photocenter;
    const double distance = std::max(this->preinit_center.norm(), 1.0);
    const double dot = old_ray.dot(new_ray) / std::max(old_ray.norm() * new_ray.norm(), 1e-9);
    const double pos_err = std::acos(std::clamp(dot, -1.0, 1.0)) * distance;
    const double yaw_err = std::abs(std::remainder(armor_yaw - this->preinit_yaw, CV_PI * 2.0)) * distance;

    if (pos_err + yaw_err > this->matcherrthresh) {
        this->InitEKF(armor_center, armor_scs, armor_yaw);
        return;
    }

    this->SavePreinit(armor_center, armor_scs, armor.photocenter, armor_yaw, armor_theta);
}

void OutPust::SavePreinit(const Eigen::Vector3d& armor_center,
                          const Eigen::Vector3d& scs,
                          const Eigen::Vector3d& photocenter,
                          double armor_yaw,
                          double armor_theta)
{
    this->has_preinit_armor = true;
    this->preinit_center = armor_center;
    this->preinit_scs = scs;
    this->preinit_photocenter = photocenter;
    this->preinit_yaw = armor_yaw;
    this->preinit_theta = armor_theta;
}

void OutPust::InitEKF(const Eigen::Vector3d& armor_center, const Eigen::Vector3d&, double armor_yaw)
{
    this->is_init = true;
    this->has_preinit_armor = false;
    this->ekfkalman.Init();

    this->yaw = std::remainder(armor_yaw, CV_PI * 2.0);
    this->w = 0.0;
    this->r = 24.0;
    this->d_h1 = 0.0;
    this->d_h2 = 0.0;

    this->center(0) = armor_center(0) - this->r * std::cos(this->yaw);
    this->center(1) = armor_center(1) - this->r * std::sin(this->yaw);
    this->center(2) = armor_center(2);
    this->UpdateArmorState();

    this->View = {ArmorView::Visual, ArmorView::Invisual, ArmorView::Invisual};
}

void OutPust::OneArmor(const ArmorPosi& armor, const Eigen::Quaterniond& gripper_to_world, double dt)
{
    MatchAns match = this->MatchErrorInEKF(armor, dt);
    const size_t id = match.id;
    const size_t side = match.side;

    Eigen::Matrix<double, 4, 1> view;
    view.block<3, 1>(0, 0) = armor.center.col(side);
    view(3, 0) = armor.yaw[side];

    for (auto& v : this->View) v = ArmorView::Invisual;
    this->View[id] = ArmorView::Visual;

    StateVector ans = this->ekfkalman.Correct(this->MakeState(), view, armor.SCS.col(side),
                                             gripper_to_world, armor.yaw_abs[side], id, dt);
    this->ApplyState(ans);
}

OutPust::MatchAns OutPust::MatchErrorInEKF(const ArmorPosi& armor, double dt) const
{
    Eigen::Vector3d predict_center;
    Eigen::Matrix<double, 4, 3> predicted = this->Predict(dt, predict_center);

    size_t farthest_id = 0;
    double farthest_distance = -1.0;
    for (size_t i = 0; i < ARMOR_NUM; ++i) {
        const double distance = (predicted.block<3, 1>(0, i) - armor.photocenter).norm();
        if (distance > farthest_distance) {
            farthest_distance = distance;
            farthest_id = i;
        }
    }

    const std::array<size_t, 2> candidates = {
        (farthest_id + 1) % ARMOR_NUM,
        (farthest_id + 2) % ARMOR_NUM
    };

    const Eigen::Vector3d robot_center_cam = predict_center - armor.photocenter;
    const Eigen::Vector3d armor_center_cam = armor.center.col(0) - armor.photocenter;
    const double robot_center_theta = std::atan2(robot_center_cam(1), robot_center_cam(0));
    const double theta_armor = std::atan2(armor_center_cam(1), armor_center_cam(0));
    const size_t side = (std::remainder(robot_center_theta - theta_armor, CV_PI * 2.0) < 0.0) ? 0 : 1;

    MatchAns best{candidates[0], side, 1e9};
    for (size_t id : candidates) {
        const double dot = predicted.block<3, 1>(0, id).dot(armor.center.col(side)) /
            std::max(predicted.block<3, 1>(0, id).norm() * armor.center.col(side).norm(), 1e-9);
        const double theta_err = std::acos(std::clamp(dot, -1.0, 1.0));
        const double yaw_err = std::abs(std::remainder(armor.yaw[side] - predicted(3, id), CV_PI * 2.0));
        const double err = theta_err + yaw_err;
        if (err < best.err) best = MatchAns{id, side, err};
    }

    return best;
}

OutPust::StateVector OutPust::MakeState() const
{
    StateVector state;
    state << this->center(0), this->center(1), this->center(2),
        this->yaw, this->w, this->r, this->d_h1, this->d_h2;
    return state;
}

void OutPust::ApplyState(const StateVector& state)
{
    this->center = state.block<3, 1>(0, 0);
    this->yaw = std::remainder(state(3, 0), CV_PI * 2.0);
    this->w = state(4, 0);
    this->r = state(5, 0);
    this->d_h1 = state(6, 0);
    this->d_h2 = state(7, 0);
    this->Speed << 0.0, 0.0, 0.0, this->w;
    this->UpdateArmorState();
}

void OutPust::UpdateArmorState()
{
    for (size_t id = 0; id < ARMOR_NUM; ++id) {
        this->Armors(0, id) = this->ArmorYaw(this->yaw, id);
        this->Armors(1, id) = this->r;
        this->Armors(2, id) = this->ArmorHeight(this->center(2), id);
    }
}

size_t OutPust::BestSide(const ArmorPosi& armor) const
{
    return armor.reproj[0] < armor.reproj[1] ? 0 : 1;
}

double OutPust::ArmorYaw(double yaw0, size_t armor_id) const
{
    return std::remainder(yaw0 + static_cast<double>(armor_id) * ARMOR_ANGLE_STEP, CV_PI * 2.0);
}

double OutPust::ArmorHeight(double z0, size_t armor_id) const
{
    return this->ArmorHeight(z0, this->d_h1, this->d_h2, armor_id);
}

double OutPust::ArmorHeight(double z0, double dh1, double dh2, size_t armor_id) const
{
    if (armor_id == 1) return z0 + dh1;
    if (armor_id == 2) return z0 + dh2;
    return z0;
}
