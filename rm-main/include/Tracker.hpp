#ifndef TRACKER_HPP
#define TRACKER_HPP
#include "Armor.hpp"
#include "Target.hpp"
#include <eigen3/Eigen/Core>

class Tracker 
{
public:
    Tracker() = default;
    
    void operator()(std::vector<ArmorPosi>& armors_posi);

    static ArmorPosi::Type type;
};


#endif // TRACKER_HPP