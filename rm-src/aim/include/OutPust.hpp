#ifndef OUT_PUST_HPP
#define OUT_PUST_HPP

#include "Armor.hpp"
#include "OPEKFKalman.hpp"

#include <array>
#include <cstddef>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <vector>

class OutPust
{
public:
    struct OutPustConfig
    {
        std::array<double, 9> gripper_to_world_matrix;
    };

    enum class ArmorView : bool
    {
        Visual = true,
        Invisual = false
    };

    explicit OutPust(const OutPustConfig& config);

    void Clear();
    bool IsInit() const { return this->is_init; }
    void ResetState(const Eigen::Vector3d& center,
                    double yaw,
                    double w,
                    double r,
                    double d_h1,
                    double d_h2);

    void Update(const std::vector<ArmorPosi>& armors, const Eigen::Quaterniond& gripper_to_world, double dt);
    void Update(double dt);

    Eigen::Matrix<double, 4, 3> Predict(double dt) const;
    Eigen::Matrix<double, 4, 3> Predict(double dt, Eigen::Vector3d& Center) const;

public:
    Eigen::Matrix<double, 3, 1> center{0.0, 0.0, 0.0};
    double yaw = 0.0;
    double w = 0.0;
    double r = 24.0;
    double d_h1 = 0.0;
    double d_h2 = 0.0;

    Eigen::Matrix<double, 4, 1> Speed{0.0, 0.0, 0.0, 0.0};
    Eigen::Matrix<double, 3, 3> Armors = Eigen::Matrix<double, 3, 3>::Zero();
    std::array<ArmorView, 3> View = {ArmorView::Invisual, ArmorView::Invisual, ArmorView::Invisual};

private:
    using StateVector = OPEKFKalman::StateVector;

    struct MatchAns
    {
        size_t id = 0;
        size_t side = 0;
        double err = 0.0;
    };

    static constexpr size_t ARMOR_NUM = 3;
    static constexpr double ARMOR_ANGLE_STEP = 2.0943951023931954923;

    OPEKFKalman ekfkalman;

    const double matcherrthresh = 100.0;

    bool is_init = false;
    bool has_preinit_armor = false;
    Eigen::Vector3d preinit_center{0.0, 0.0, 0.0};
    Eigen::Vector3d preinit_scs{0.0, 0.0, 0.0};
    Eigen::Vector3d preinit_photocenter{0.0, 0.0, 0.0};
    double preinit_yaw = 0.0;
    double preinit_theta = 0.0;

    void TryInitByJump(const ArmorPosi& armor, size_t side);
    void SavePreinit(const Eigen::Vector3d& armor_center,
                     const Eigen::Vector3d& scs,
                     const Eigen::Vector3d& photocenter,
                     double armor_yaw,
                     double armor_theta);
    void InitEKF(const Eigen::Vector3d& armor_center, const Eigen::Vector3d& scs, double armor_yaw);

    void OneArmor(const ArmorPosi& armor, const Eigen::Quaterniond& gripper_to_world, double dt);
    MatchAns MatchErrorInEKF(const ArmorPosi& armor, double dt) const;

    StateVector MakeState() const;
    void ApplyState(const StateVector& state);
    void UpdateArmorState();

    size_t BestSide(const ArmorPosi& armor) const;
    double ArmorYaw(double yaw0, size_t armor_id) const;
    double ArmorHeight(double z0, size_t armor_id) const;
    double ArmorHeight(double z0, double dh1, double dh2, size_t armor_id) const;
};

#endif // OUT_PUST_HPP
