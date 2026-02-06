#include "../include/EKFKalman.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>

void EKFKalman::Init()
{
    this->CovState = this->CovStateInit;
};


Eigen::Matrix<double, 8, 1> EKFKalman::operator()
    (
        const Eigen::Matrix<double, 8, 1>& State,
        const Eigen::Matrix<double, 4, 4>& CovArmor, 
        const Eigen::Matrix<double, 4, 1>& View,
        const Eigen::Matrix<double, 3, 1>& Center, 
        double dt,
        double err
    )
{
    
};

