#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <eigen3/Eigen/Core>
#include <optional>

#include "../include/TargetState.hpp"
#include "tinympc/tiny_api.hpp"
#include "BulletTrajectory.hpp"

namespace MPC
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control;
  bool fire;
  float target_yaw;
  float target_pitch;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
};

class Planner
{
public:

  struct PlannerConfig
  {
   double yaw_offset_;
   double pitch_offset_;
   double fire_thresh_;
   double decision_speed_;
   double high_speed_delay_time_;
   double low_speed_delay_time_;
   double max_yaw_acc_;
   std::vector<double> Q_yaw_;
   std::vector<double> R_yaw_;
   double max_pitch_acc_;
   std::vector<double> Q_pitch_;
   std::vector<double> R_pitch_;
  };

  Eigen::Vector4d debug_xyza;

  explicit Planner(const std::string & config_path);
  explicit Planner(const PlannerConfig& config);

  Plan plan(RobotState& target, double bullet_speed);
  Plan plan(const std::unique_ptr<RobotState>& target_ptr, double bullet_speed);

private:
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  void setup_yaw_solver(double max_yaw_acc,const std::vector<double>& Q_yaw, const std::vector<double>& R_yaw );
  void setup_pitch_solver(double max_pitch_acc,const std::vector<double>& Q_pitch, const std::vector<double>& R_pitch );

  Eigen::Matrix<double, 2, 1> aim(const Eigen::Matrix<double, 4, 4>& armors_posi, double bullet_speed);
  Trajectory get_trajectory(const RobotState & target, double yaw0, double bullet_speed);
};

}  // namespace MPC

#endif  // AUTO_AIM__PLANNER_HPP