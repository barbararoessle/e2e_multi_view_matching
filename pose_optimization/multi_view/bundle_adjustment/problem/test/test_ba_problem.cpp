#include <fstream>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <ba_problem.h>
#include <file_utils.h>

std::vector<std::array<double, 3>> Project(const std::vector<std::array<double, 3>>& pts_3d, 
    const double* const extr, const double* const intr)
{
    double f_x{intr[0]}, f_y{intr[1]}, c_x{intr[2]}, c_y{intr[3]};
    std::vector<std::array<double, 3>> pts_2d{};
    for (const auto& pt_3d : pts_3d)
    {
        double pt_3d_cam[3];
        ceres::AngleAxisRotatePoint(extr, pt_3d.data(), pt_3d_cam);
        pt_3d_cam[0] += extr[3];
        pt_3d_cam[1] += extr[4];
        pt_3d_cam[2] += extr[5];

        pts_2d.emplace_back(std::array<double, 3>{pt_3d_cam[0] * f_x / pt_3d_cam[2] + c_x, 
            pt_3d_cam[1] * f_y / pt_3d_cam[2] + c_y, pt_3d_cam[2]});
    }
    return pts_2d;
}

double Err(double max_err)
{
    return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX) * 2. * max_err - max_err;
}

void DefineProblem(const double* const extr_1, double err_cam = 0., double err_2d = 0., double err_3d = 0., 
    bool init = true)
{
    std::srand(0);
    int n_cams{2};
    double extr_0[] = {0., 0., 0., 0., 0., 0.}; // world -> cam

    double intr[] = {1., 1., 0., 0.};//{575., 575., 320., 240.};
    // convert err_2d in pixels to normalized image coordinates
    double assumed_focal_length = 575.;
    err_2d /= assumed_focal_length;

    const std::vector<std::array<double, 3>> pts_3d{
        {-2., 1., 1.}, {-1., 0., 1.5}, {0., 2., 1.}, {1., 0.5, 1.5}, {2., -1., 1.}};

    const auto& pts_2d_0 = Project(pts_3d, extr_0, intr);
    const auto& pts_2d_1 = Project(pts_3d, extr_1, intr);

    std::ofstream f("ba_in.csv");
    
    // write header
    f << std::setprecision(12) << n_cams << "," << 0 << "," << pts_3d.size() << "," << pts_2d_0.size() + pts_2d_1.size() << ","
        << intr[0] << "," << intr[1] << "," << intr[2] << "," << intr[3] << "\n";
    
    // write 2d observations
    int i_cams{0};
    for (const auto& pts_2d : {pts_2d_0, pts_2d_1})
    {
        int i_pts_3d{0};
        for (const auto& pt_2d : pts_2d)
        {
            f << i_cams << "," << i_pts_3d << "," << pt_2d[0] + Err(err_2d) << "," << pt_2d[1] + Err(err_2d) << "," 
                << pt_2d[2] << "\n";
            ++i_pts_3d;
        }
        ++i_cams;
    }

    // write cameras
    int n_zero_cams{2};
    if (init)
    {
        n_zero_cams = 1;
    }
    for (int i{0}; i != n_zero_cams; ++i)
    {
        f << 0. << "," << 0. << "," << 0. << "," << 0. << "," << 0. << "," << 0. << "," << 0. << "," << 0. << "," << 0. << "," 
            << 0. << "," << 0. << "," << 0. << "\n";
    }
    if (init)
    {
        double extr_1_err[] = {extr_1[0] + Err(err_cam), extr_1[1] + Err(err_cam), extr_1[2] + Err(err_cam), 
            extr_1[3] + Err(err_cam), extr_1[4] + Err(err_cam), extr_1[5] + Err(err_cam)};

        double R_err[9];
        ceres::AngleAxisToRotationMatrix(extr_1_err, R_err);
        for (int i{0}; i != 9; ++i)
        {
            f << R_err[i] << ",";
        }
        f << extr_1_err[3] << "," << extr_1_err[4] << "," << extr_1_err[5] << "\n";
    }

    // write 3d points
    for (const auto& pt_3d : pts_3d)
    {
        if (init)
        {
            f << pt_3d[0] + Err(err_3d) << "," << pt_3d[1] + Err(err_3d) << "," << pt_3d[2] + Err(err_3d) << "\n";
        }
        else
        {
            f << 0. << "," << 0. << "," << 0. << "\n";
        }
    }
}

std::vector<std::array<double, 12>> ReadResult(const std::string& file_path)
{
    std::vector<std::array<double, 12>> result{};
    std::ifstream file(file_path);
    std::string line{};
    while (std::getline(file, line))
    {
        const auto& elements = SplitByChar(line, ',');
        std::array<double, 12> R_t{};
        for (int i{0}; i != 12; ++i)
        {
            R_t[i] = std::stod(elements[i]);
        }
        result.emplace_back(R_t);
    }
    return result;
}

void ExpectArrayEq(const double* ptr_0, const double* ptr_1, int count, double err)
{
    for (int i{0}; i != count; ++i, ++ptr_0, ++ptr_1)
    {
        EXPECT_NEAR(*ptr_0, *ptr_1, err);
    }
}

void SolveAndCheckResult(const double* const expected_extr, int n_dims = 2, double max_err = 1e-6)
{
    BaProblem ba_problem("ba_in.csv");
    Solve(ba_problem);
    std::string out_file("ba_out.csv");
    ba_problem.WriteResult(out_file);
    const auto& cams = ReadResult(out_file);
    const auto& cam = cams[0];
    for (int i{0}; i != cams.size(); ++i)
    {
        const auto& cam = cams[i];
        double extr[] = {0., 0., 0., cam[9], cam[10], cam[11]};
        ceres::RotationMatrixToAngleAxis(cam.data(), extr);
        // 0th camera is fixed at 0s
        if (i == 0)
        {
            const double zeros[] = {0., 0., 0., 0., 0., 0.};
            ExpectArrayEq(zeros, extr, 6, 1e-6);
        }
        // check other camera
        else
        {
            ExpectArrayEq(expected_extr, extr, 6, max_err);
        }
    }
}

TEST(Reprojection2d, Perfect2Cams5Pts)
{
    double expected_extr_1[] = {0.3, -0.2, 0.5, 0.3, -0.4, 0.5};
    DefineProblem(expected_extr_1);
    SolveAndCheckResult(expected_extr_1);
}

TEST(Reprojection2d, Noisy2Cams5Pts)
{
    double expected_extr_1[] = {0.3, -0.2, 0.5, 0.3, -0.4, 0.5};
    DefineProblem(expected_extr_1, 0.1, 10., 0.2);
    SolveAndCheckResult(expected_extr_1, 2, 9e-2);
}

TEST(Reprojection2d, MoreNoisy2Cams5Pts)
{
    double expected_extr_1[] = {0.3, -0.2, 0.5, 0.3, -0.4, 0.5};
    DefineProblem(expected_extr_1, 0.2, 0., 0.3);
    SolveAndCheckResult(expected_extr_1, 2, 4e-2);
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}