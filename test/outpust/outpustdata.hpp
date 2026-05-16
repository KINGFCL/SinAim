#ifndef TEST_OUTPUST_OUTPUSTDATA_HPP
#define TEST_OUTPUST_OUTPUSTDATA_HPP

#include "Armor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <opencv2/core/cvdef.h>
#include <vector>

namespace outpust_test
{

constexpr double kYawStep = 2.0 * CV_PI / 3.0;
constexpr double kVisibleLimit = 80.0 * CV_PI / 180.0;

inline double Wrap(double angle)
{
    return std::remainder(angle, 2.0 * CV_PI);
}

struct SolverLikeObservationConfig
{
    Eigen::Vector3d photocenter_world{0.0, 0.0, 0.0};
    Eigen::Matrix3d r_world_to_cam = Eigen::Matrix3d::Identity();
    double good_reproj = 0.05;
    double bad_reproj = 2.0;
    bool output_all_visible = false;
};

struct SyntheticOutpust
{
    Eigen::Vector3d center{500.0, 0.0, 15.0};
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
            armors(2, id) = this->ArmorHeight(id);
            armors(3, id) = yaw;
        }

        return armors;
    }

    double ArmorHeight(int id) const
    {
        if (id == 1) return this->center(2) + this->d_h1;
        if (id == 2) return this->center(2) + this->d_h2;
        return this->center(2);
    }

    bool Visible(const Eigen::Matrix<double, 4, 3>& armors,
                 int id,
                 const Eigen::Vector3d& photocenter_world) const
    {
        Eigen::Vector3d sight = armors.block<3, 1>(0, id) - photocenter_world;
        sight(2) = 0.0;
        if (sight.norm() < 1e-9) return false;
        sight.normalize();

        const double yaw = armors(3, id);
        const Eigen::Vector3d front_normal{-std::cos(yaw), -std::sin(yaw), 0.0};
        return front_normal.dot(sight) >= std::cos(kVisibleLimit);
    }

    int FirstVisibleId(double t, const Eigen::Vector3d& photocenter_world) const
    {
        const Eigen::Matrix<double, 4, 3> armors = this->Truth(t);
        for (int id = 0; id < 3; ++id) {
            if (this->Visible(armors, id, photocenter_world)) return id;
        }
        return -1;
    }

    ArmorPosi MakeSolverLikeArmorPosi(const Eigen::Matrix<double, 4, 3>& armors,
                                      int id,
                                      const SolverLikeObservationConfig& cfg) const
    {
        const Eigen::Vector3d p_world = armors.block<3, 1>(0, id);
        const Eigen::Vector3d p_cam = cfg.r_world_to_cam * (p_world - cfg.photocenter_world);
        const double yaw = Wrap(armors(3, id));
        const Eigen::Vector3d p_base = p_world - cfg.photocenter_world;
        const double center_theta = std::atan2(p_base(1), p_base(0));
        const double mirrored_yaw = Wrap(2.0 * center_theta - yaw);

        const Eigen::Vector3d toward_world{-std::cos(yaw), -std::sin(yaw), 0.0};
        const double cross2d = p_base(0) * toward_world(1) - p_base(1) * toward_world(0);
        const int good_side = (cross2d < 0.0) ? 0 : 1;
        const int bad_side = 1 - good_side;

        Eigen::Matrix<double, 3, 2> centers_world = Eigen::Matrix<double, 3, 2>::Zero();
        Eigen::Matrix<double, 3, 2> centers_cam = Eigen::Matrix<double, 3, 2>::Zero();
        std::array<double, 2> yaws{};
        std::array<double, 2> reproj{};

        centers_world.col(good_side) = p_world;
        centers_cam.col(good_side) = p_cam;
        yaws[good_side] = yaw;
        reproj[good_side] = cfg.good_reproj;

        centers_world.col(bad_side) = p_world;
        centers_cam.col(bad_side) = p_cam;
        yaws[bad_side] = mirrored_yaw;
        reproj[bad_side] = cfg.bad_reproj;

        ArmorPosi armor(centers_world, centers_cam, cfg.photocenter_world, yaws, reproj, true);
        armor.type = ArmorPosi::Type::outpost;
        armor.confidence = 1.0f;
        return armor;
    }

    std::vector<ArmorPosi> SolverLikeObservations(
        double t,
        const SolverLikeObservationConfig& cfg = SolverLikeObservationConfig{}) const
    {
        const Eigen::Matrix<double, 4, 3> armors = this->Truth(t);
        std::vector<ArmorPosi> observations;

        for (int id = 0; id < 3; ++id) {
            if (!this->Visible(armors, id, cfg.photocenter_world)) continue;
            observations.emplace_back(this->MakeSolverLikeArmorPosi(armors, id, cfg));
            if (!cfg.output_all_visible) break;
        }

        return observations;
    }
};

inline double MaxPositionError(const Eigen::Matrix<double, 4, 3>& a,
                               const Eigen::Matrix<double, 4, 3>& b)
{
    double max_error = 0.0;
    for (int i = 0; i < 3; ++i) {
        max_error = std::max(max_error, (a.block<3, 1>(0, i) - b.block<3, 1>(0, i)).norm());
    }
    return max_error;
}

inline double MinAssignmentPositionError(const Eigen::Matrix<double, 4, 3>& truth,
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

}  // namespace outpust_test

#endif  // TEST_OUTPUST_OUTPUSTDATA_HPP
