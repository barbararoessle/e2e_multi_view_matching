#include <iomanip>
#include <fstream>
#include <cassert>

#include <ceres/rotation.h>

#include <ba_init.h>
#include <file_utils.h>

BaInit::BaInit(const std::string& file_path)
{
    std::ifstream file(file_path);
    std::string line{};
    while (std::getline(file, line))
    {
        const auto& elements = SplitByChar(line, ',');
        if (elements.size() == 10)
        {
            ++n_views_;
            const auto view_id = std::stoi(elements[0]);
            double R[9];
            Eigen::Vector3d angle_axis = Eigen::Vector3d::Zero();
            for (int i{0}; i != 9; ++i)
            {
                R[i] = std::stod(elements[i + 1]);
            }
            ceres::RotationMatrixToAngleAxis(R, angle_axis.data()); // initialize rotation
            global_rotations_[view_id] = angle_axis;
        }
        else if (elements.size() == 14) // 3d points have 3 elements
        {
            const auto view_id0 = std::stoi(elements[0]);
            const auto view_id1 = std::stoi(elements[1]);
            double R[9];
            Eigen::Vector3d angle_axis = Eigen::Vector3d::Zero();
            Eigen::Vector3d t = Eigen::Vector3d::Zero();
            for (int i{0}; i != 9; ++i)
            {
                R[i] = std::stod(elements[i + 2]);
            }
            for (int i{0}; i != 3; ++i)
            {
                t[i] = std::stod(elements[i + 11]);
            }
            ceres::RotationMatrixToAngleAxis(R, angle_axis.data());
            theia::TwoViewInfo vi{};
            vi.rotation_2 = angle_axis;
            vi.position_2 = t;
            view_pairs_[theia::ViewIdPair{view_id0, view_id1}] = vi;
        }
    }
}

BaInit::~BaInit()
{
}

void BaInit::WriteResult(const std::string& file_path) const
{
    std::ofstream file(file_path);
    for (int id{0}; id != n_views_; ++id)
    {
        const auto& rot = global_rotations_.at(id);
        const auto& transl = global_translations_.at(id);
        // convert from position vector to extrinsics translation vector
        const auto& t_extr = -Eigen::AngleAxisd(rot.norm(), rot.normalized()).toRotationMatrix() * transl;
        double R[] = {0., 0., 0., 0., 0., 0., 0., 0., 0.}; // column major
        ceres::AngleAxisToRotationMatrix(rot.data(), R);
        for (int i{0}; i != 9; ++i)
        {
            file << std::setprecision(12) << R[i] << ",";
        }
        file << std::setprecision(12) << t_extr[0] << "," << t_extr[1] << "," << t_extr[2] << "\n";
    }
}

void BaInit::Run()
{
    theia::RobustRotationEstimator rotation_estimator(theia::RobustRotationEstimator::Options{});
    bool success = rotation_estimator.EstimateRotations(view_pairs_, &global_rotations_);
    if (!success)
    {
        std::cout << "EstimateRotations failed." << std::endl;
    }
    theia::LeastUnsquaredDeviationPositionEstimator position_estimator(theia::LeastUnsquaredDeviationPositionEstimator::Options{});
    success = position_estimator.EstimatePositions(view_pairs_, global_rotations_, &global_translations_);
    if (!success)
    {
        std::cout << "EstimatePositions failed." << std::endl;
    }
}