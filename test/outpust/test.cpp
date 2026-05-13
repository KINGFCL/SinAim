#include "OutPust.hpp"
#include "RerunVisualizer.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <opencv2/core/cvdef.h>
#include <string>
#include <thread>
#include <vector>

namespace
{
constexpr double kYawStep = 2.0 * CV_PI / 3.0;
constexpr double kVisibleLimit = 80.0 * CV_PI / 180.0;

double Wrap(double angle)
{
    return std::remainder(angle, 2.0 * CV_PI);
}

struct SyntheticOutpost
{
    Eigen::Vector3d center{500.0, 0.0, 150.0};
    double yaw0 = CV_PI;
    double w = 5.0;
    double r = 25.0;
    double d_h1 = 10.0;
    double d_h2 = 20.0;

    Eigen::Matrix<double, 4, 3> Truth(double t) const
    {
        Eigen::Matrix<double, 4, 3> armors;
        const double base_yaw = Wrap(this->yaw0 + this->w * t);

        for (int id = 0; id < 3; ++id) {
            const double yaw = Wrap(base_yaw + id * kYawStep);
            armors(0, id) = this->center(0) + this->r * std::cos(yaw);
            armors(1, id) = this->center(1) + this->r * std::sin(yaw);
            armors(2, id) = this->center(2) + (id == 1 ? this->d_h1 : (id == 2 ? this->d_h2 : 0.0));
            armors(3, id) = yaw;
        }

        return armors;
    }

    bool Visible(const Eigen::Matrix<double, 4, 3>& armors, int id) const
    {
        Eigen::Vector3d los{armors(0, id), armors(1, id), 0.0};
        if (los.norm() < 1e-9) return false;
        los.normalize();

        Eigen::Vector3d normal{std::cos(armors(3, id)), std::sin(armors(3, id)), 0.0};
        const double face_proj = -normal.dot(los);
        return face_proj >= std::cos(kVisibleLimit);
    }

    ArmorPosi MakeObservation(const Eigen::Matrix<double, 4, 3>& armors, int id) const
    {
        ArmorPosi armor;
        const Eigen::Vector3d p = armors.block<3, 1>(0, id);
        const double yaw = armors(3, id);
        const double norm = p.norm();
        const double xy = p.head<2>().norm();
        const Eigen::Vector3d scs{norm, std::asin(xy / std::max(norm, 1e-9)), std::atan2(p(1), p(0))};
        const double theta = std::atan2(p(1), p(0));
        const double yaw_abs = std::abs(Wrap(yaw + CV_PI - theta));

        armor.center.col(0) = p;
        armor.center.col(1) = p;
        armor.photocenter.setZero();
        armor.yaw = {yaw, yaw};
        armor.reproj = {0.0, 1.0};
        armor.theta = {theta, theta};
        armor.yaw_abs = {yaw_abs, yaw_abs};
        armor.SCS.col(0) = scs;
        armor.SCS.col(1) = scs;
        armor.IsInRange = true;
        armor.type = ArmorPosi::Type::outpost;
        armor.confidence = 1.0f;

        return armor;
    }

    std::vector<ArmorPosi> Observations(double t) const
    {
        const auto armors = this->Truth(t);
        std::vector<ArmorPosi> observations;

        for (int id = 0; id < 3; ++id) {
            if (this->Visible(armors, id)) {
                observations.emplace_back(this->MakeObservation(armors, id));
                break;
            }
        }

        return observations;
    }

    int FirstVisibleId(double t) const
    {
        const auto armors = this->Truth(t);
        for (int id = 0; id < 3; ++id) {
            if (this->Visible(armors, id)) return id;
        }
        return -1;
    }
};

double MaxPositionError(const Eigen::Matrix<double, 4, 3>& a, const Eigen::Matrix<double, 4, 3>& b)
{
    double max_error = 0.0;
    for (int i = 0; i < 3; ++i) {
        max_error = std::max(max_error, (a.block<3, 1>(0, i) - b.block<3, 1>(0, i)).norm());
    }
    return max_error;
}

double MinAssignmentPositionError(const Eigen::Matrix<double, 4, 3>& truth,
                                  const Eigen::Matrix<double, 4, 3>& modeled)
{
    double max_error = 0.0;
    for (int i = 0; i < 3; ++i) {
        double best = 1e9;
        for (int j = 0; j < 3; ++j) {
            best = std::min(best, (truth.block<3, 1>(0, i) - modeled.block<3, 1>(0, j)).norm());
        }
        max_error = std::max(max_error, best);
    }
    return max_error;
}
}  // namespace

int main()
{
    SyntheticOutpost truth_model;

    OutPust::OutPustConfig cfg{{1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0}};
    OutPust tracker(cfg);
    tracker.ResetState(truth_model.center, truth_model.yaw0, truth_model.w,
                       truth_model.r, truth_model.d_h1, truth_model.d_h2);

    RerunVisualizer viz(std::string("OutPust_Unit_Test"));
    Eigen::Quaterniond gripper_to_world = Eigen::Quaterniond::Identity();

    constexpr double dt = 0.02;
    constexpr int steps = 600;

    for (int step = 0; step < steps; ++step) {
        const double t = (step + 1) * dt;
        const Eigen::Matrix<double, 4, 3> truth = truth_model.Truth(t);
        std::vector<ArmorPosi> observations = truth_model.Observations(t);
        const int visible_id = truth_model.FirstVisibleId(t);

        tracker.Update(observations, gripper_to_world, dt);
        const Eigen::Matrix<double, 4, 3> modeled = tracker.Predict(0.0);
        const double max_pos_error = MaxPositionError(truth, modeled);
        const double assignment_error = MinAssignmentPositionError(truth, modeled);
        const double yaw_error = std::abs(Wrap(tracker.yaw - Wrap(truth_model.yaw0 + truth_model.w * t)));

        if (step % 25 == 0) {
            std::cout << "step=" << step
                      << " obs=" << observations.size()
                      << " visible_id=" << visible_id
                      << " max_pos_error_cm=" << max_pos_error
                      << " assignment_error_cm=" << assignment_error
                      << " yaw_error_rad=" << yaw_error
                      << " w=" << tracker.w
                      << " r=" << tracker.r
                      << " dh1=" << tracker.d_h1
                      << " dh2=" << tracker.d_h2
                      << '\n';
        }

        viz.outpust(tracker, truth, observations, dt);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    return 0;
}
