#include <iomanip>
#include <fstream>
#include <cassert>

#include <ba_problem.h>
#include <file_utils.h>

BaProblem::BaProblem(const std::string& file_path)
    : n_dims_(2)
{
    std::ifstream file(file_path);
    std::string line{};
    int cams_count{0}, obs_count{0}, pts_count{0};
    while (std::getline(file, line))
    {
        const auto& elements = SplitByChar(line, ',');
        if (elements.size() == 8)
        {
            n_cams_ = std::stoi(elements[0]);
            fixed_cam_id_ = std::stoi(elements[1]);
            n_pts_ = std::stoi(elements[2]);
            n_obs_ = std::stoi(elements[3]);
            point_index_ = new int[n_obs_];
            camera_index_ = new int[n_obs_];
            observations_ = new double[n_dims_ * n_obs_];
            observation_weights_ = new double[n_dims_ * n_obs_];
            n_params_ = 6 * n_cams_ + 3 * n_pts_;
            parameters_ = new double[n_params_];
            f_x_ = std::stod(elements[4]);
            f_y_ = std::stod(elements[5]);
            c_x_ = std::stod(elements[6]);
            c_y_ = std::stod(elements[7]);
        }
        else if (elements.size() == 3) // 3d points have 3 elements
        {
            double* const it_3d_pt = mutable_points() + 3 * pts_count;
            for (int i{0}; i != 3; ++i)
            {
                *(it_3d_pt + i) = std::stod(elements[i]);
            }
            ++pts_count;
        }
        else if (elements.size() >= 4 && elements.size() <= 6) // observations have 5/6 elements
        {
            int cam_idx = std::stoi(elements[0]);
            int pt_idx = std::stoi(elements[1]);
            double x = std::stod(elements[2]);
            double y = std::stod(elements[3]);
            *(camera_index_ + obs_count) = cam_idx;
            *(point_index_ + obs_count) = pt_idx;
            *(observations_ + n_dims_ * obs_count) = x;
            *(observations_ + n_dims_ * obs_count + 1) = y;
            double w_x(1.), w_y(1.);
            if (elements.size() == 5)
            {
                w_x = w_y = std::stod(elements[4]);
            }
            else if (elements.size() == 6)
            {
                w_x = std::stod(elements[4]);
                w_y = std::stod(elements[5]);
            }
            *(observation_weights_ + n_dims_ * obs_count) = w_x;
            *(observation_weights_ + n_dims_ * obs_count + 1) = w_y;
            
            ++obs_count;
        }

        else if (elements.size() == 12) // cameras have 12 elements
        {
            double R[9];
            double* const it_cam = mutable_cameras() + 6 * cams_count;
            for (int i{0}; i != 12; ++i)
            {
                if (i < 9)
                {
                    R[i] = std::stod(elements[i]);
                }
                else
                {
                    it_cam[i - 6] = std::stod(elements[i]); // initialize translation
                }
            }
            ceres::RotationMatrixToAngleAxis(R, it_cam); // initialize rotation
            ++cams_count;
        }
    }
}

BaProblem::~BaProblem()
{
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
}

void BaProblem::WriteResult(const std::string& file_path) const
{
    std::ofstream file(file_path);
    for (int i{0}; i != n_cams_; ++i)
    {
        const double* const it_param = parameters_ + i * 6;
        double angle_axis[] = {*(it_param), *(it_param + 1), *(it_param + 2)};
        double R[] = {0., 0., 0., 0., 0., 0., 0., 0., 0.}; // column major
        ceres::AngleAxisToRotationMatrix(angle_axis, R);
        for (int r{0}; r != 9; ++r)
        {
            file << std::setprecision(12) << R[r] << ",";
        }
        file << *(it_param + 3) << "," << *(it_param + 4) << "," << *(it_param + 5) << "\n";
    }
}

void Solve(BaProblem& ba_problem)
{
    const auto n_dims = ba_problem.NumberDimensions();
    const double* observations = ba_problem.observations();
    const double* observation_weights = ba_problem.observation_weights();
    const double f_x = ba_problem.GetFx();
    const double f_y = ba_problem.GetFy();
    const double c_x = ba_problem.GetCx();
    const double c_y = ba_problem.GetCy();

    // create residuals for each observation in the bundle adjustment problem
    ceres::Problem problem;
    for (int i = 0; i < ba_problem.num_observations(); ++i)
    {
        if (ba_problem.is_observed_by_fixed_camera(i))
        {
            ceres::CostFunction* cost_function = ReprojectionErrorFixedCamera::Create(
                observations[n_dims * i + 0], observations[n_dims * i + 1], f_x, f_y, c_x, c_y, 
                observation_weights[n_dims * i + 0], observation_weights[n_dims * i + 1]);
            problem.AddResidualBlock(cost_function,
                                    NULL /* squared loss */,
                                    ba_problem.mutable_point_for_observation(i));
        }
        else
        {
            ceres::CostFunction* cost_function = ReprojectionError::Create(
                observations[n_dims * i + 0], observations[n_dims * i + 1], f_x, f_y, c_x, c_y, 
                observation_weights[n_dims * i + 0], observation_weights[n_dims * i + 1]);
            problem.AddResidualBlock(cost_function,
                                    NULL /* squared loss */,
                                    ba_problem.mutable_camera_for_observation(i),
                                    ba_problem.mutable_point_for_observation(i));
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}