#include "OutPust.hpp"
#include "RerunVisualizer.hpp"
#include "outpustdata.hpp"

#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <opencv2/core/cvdef.h>
#include <string>
#include <thread>
#include <vector>

int main()
{
    outpust_test::SyntheticOutpust truth_model;
    outpust_test::SolverLikeObservationConfig observation_cfg;
    observation_cfg.output_all_visible = false;

    OutPust::OutPustConfig cfg{{1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0,
                                0.0, 0.0, 1.0}};
    OutPust outpust(cfg);
    OutPust* current_outpust = nullptr;

    RerunVisualizer viz(std::string("OutPust_Unit_Test"));
    Eigen::Quaterniond gripper_to_world = Eigen::Quaterniond::Identity();

    constexpr double dt = 0.02;
    constexpr int steps = 600;

    for (int step = 0; step < steps; ++step) {
        const double t = (step + 1) * dt;
        const Eigen::Matrix<double, 4, 3> truth = truth_model.Truth(t);
        std::vector<ArmorPosi> armors = truth_model.SolverLikeObservations(t, observation_cfg);
        const int visible_id = truth_model.FirstVisibleId(t, observation_cfg.photocenter_world);

        outpust.Update(armors, gripper_to_world, dt);
        current_outpust = outpust.IsInit() ? &outpust : nullptr;

        if (current_outpust != nullptr) {
            const Eigen::Matrix<double, 4, 3> modeled = current_outpust->Predict(0.0);
            const double max_pos_error = outpust_test::MaxPositionError(truth, modeled);
            const double assignment_error = outpust_test::MinAssignmentPositionError(truth, modeled);
            const double truth_yaw = outpust_test::Wrap(truth_model.yaw0 + truth_model.w * t);
            const double yaw_error = std::abs(outpust_test::Wrap(current_outpust->yaw - truth_yaw));

            if (step % 25 == 0) {
                std::cout << "step=" << step
                          << " obs=" << armors.size()
                          << " visible_id=" << visible_id
                          << " max_pos_error_cm=" << max_pos_error
                          << " assignment_error_cm=" << assignment_error
                          << " yaw_error_rad=" << yaw_error
                          << " w=" << current_outpust->w
                          << " r=" << current_outpust->r
                          << " dh1=" << current_outpust->d_h1
                          << " dh2=" << current_outpust->d_h2
                          << '\n';
            }

            viz.outpust(*current_outpust, current_outpust->Predict(0), armors, dt);
        } else if (step % 25 == 0) {
            std::cout << "step=" << step
                      << " obs=" << armors.size()
                      << " visible_id=" << visible_id
                      << " waiting_for_outpust_init\n";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    return 0;
}
