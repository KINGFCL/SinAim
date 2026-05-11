#include "Eigen/Core"
#include "Eigen/Geometry"
#include <array>

class OutPustKalman
{
public:
    OutPustKalman(const Eigen::Matrix3d& RCamera2Grip);: RCamera2Grip(RCamera2Grip) {}
    ~OutPustKalman();


    //状态量只有 
};