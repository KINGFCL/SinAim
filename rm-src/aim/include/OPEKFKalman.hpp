#ifndef OPEKFKALMAN_HPP
#define OPEKFKALMAN_HPP

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

class OPEKFKalman
{
public:
    using StateVector = Eigen::Matrix<double, 8, 1>;
    using StateMatrix = Eigen::Matrix<double, 8, 8>;

    explicit OPEKFKalman(const Eigen::Matrix3d& RCamera2Grip);

    void Init();

    StateVector Predict(const StateVector& state, double dt);

    StateVector Correct(const StateVector& state,
                        const Eigen::Matrix<double, 4, 1>& view,
                        const Eigen::Vector3d& scs,
                        const Eigen::Quaterniond& gripper_to_world,
                        double delta_angle,
                        size_t armor_id,
                        double dt);

    Eigen::Matrix<double, 4, 1> StateToView(const StateVector& state, size_t armor_id) const;

    Eigen::Matrix<double, 4, 8> GetStateToViewJacobian(const StateVector& state, size_t armor_id) const;

    StateMatrix CovState;

private:
    static constexpr size_t ARMOR_NUM = 3;
    static constexpr double ARMOR_ANGLE_STEP = 2.0943951023931954923;

    Eigen::Matrix3d RCamera2Grip;

    const double Max_Robust_Scale = 2.0;
    const double Threshold_Angle = 0.8;
    const double Var_r = 100.0;
    const double Var_yaw = 0.003;
    const double Var_a_z = 100.0;
    const double Var_alpha = 10.0;
    const double Var_radius = 0.01;
    const double Var_height_diff = 0.01;

    Eigen::Matrix3d GetJacobianSphericalToCartesian(const Eigen::Vector3d& scs) const;
    double GetAngleRobustScale(double angle_error) const;
    double ArmorYaw(double yaw0, size_t armor_id) const;
    double ArmorHeight(double z0, double dh1, double dh2, size_t armor_id) const;
};

#endif // OPEKFKALMAN_HPP
