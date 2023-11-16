#include <fstream>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <ceres/rotation.h>
#include <theia/theia.h>

#include <ba_init.h>

double Err(double max_err)
{
    return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX) * 2. * max_err - max_err;
}

std::unordered_map<theia::ViewIdPair, theia::TwoViewInfo> CreateViewPairs(const std::vector<Eigen::Affine3d>& extrinsics, 
    double max_err = 0.)
{
    std::unordered_map<theia::ViewIdPair, theia::TwoViewInfo> view_pairs{};
    for (int id1{0}; id1 != extrinsics.size(); ++id1)
    {
        for (int id0{0}; id0 != id1; ++id0)
        {
            theia::TwoViewInfo vi{};
            const auto& T_021 = extrinsics[id1] * extrinsics[id0].inverse();
            const auto& R_021 = Eigen::AngleAxisd(T_021.rotation());
            vi.rotation_2 = R_021.axis() * R_021.angle();
            vi.rotation_2 += Eigen::Vector3d{Err(max_err), Err(max_err), Err(max_err)};
            vi.position_2 = T_021.inverse().translation();
            vi.position_2 += Eigen::Vector3d{Err(max_err), Err(max_err), Err(max_err)};
            view_pairs[theia::ViewIdPair{id0, id1}] = vi;
        }
    }
    return view_pairs;
}

std::unordered_map<theia::ViewId, Eigen::Vector3d> GetGlobalRotations(const std::vector<Eigen::Affine3d>& extrinsics, 
    double max_err = 0.)
{
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations{};
    for (std::uint32_t id{0}; id != extrinsics.size(); ++id)
    {
        const auto& R_id = Eigen::AngleAxisd(extrinsics[id].rotation());
        estimated_rotations[id] = R_id.axis() * R_id.angle();
        estimated_rotations[id] += Eigen::Vector3d{Err(max_err), Err(max_err), Err(max_err)};
    }
    return estimated_rotations;
}

void ExpectRotationsEqual(const std::vector<Eigen::Affine3d>& exp_extrinsics, 
    const std::unordered_map<theia::ViewId, Eigen::Vector3d>& angle_axis, double err = 1e-6, bool verbose = false)
{
    for (int i{0}; i != exp_extrinsics.size(); ++i)
    {
        const auto& exp_rot = Eigen::AngleAxisd(exp_extrinsics[i].rotation());
        const auto& exp_angle_axis = exp_rot.angle() * exp_rot.axis();
        if (verbose)
        {
            std::cout << exp_angle_axis  << std::endl << std::endl << angle_axis.at(i) << "\n\n---\n\n";
        }
        for (int j{0}; j != 3; ++j)
        {
            EXPECT_NEAR(exp_angle_axis[j], angle_axis.at(i)[j], err);
        }
    }
}

void ExpectTranslationsEqual(const std::vector<Eigen::Affine3d>& exp_extrinsics, 
    const std::unordered_map<theia::ViewId, Eigen::Vector3d>& translations, double err = 1e-6, bool verbose = false)
{
    for (int i{0}; i != exp_extrinsics.size(); ++i)
    {
        const auto& exp_trans = exp_extrinsics[i].inverse().translation();
        if (verbose)
        {
            std::cout << exp_trans  << std::endl << std::endl << translations.at(i) << "\n\n---\n\n";
        }
        for (int j{0}; j != 3; ++j)
        {
            EXPECT_NEAR(exp_trans[j], translations.at(i)[j], err);
        }
    }
}

std::vector<Eigen::Affine3d> CreateCameraExtrinsics()
{
    return std::vector<Eigen::Affine3d>{
        (Eigen::Translation3d(Eigen::Vector3d{0., 0., 0.}) * Eigen::AngleAxisd(0., Eigen::Vector3d{0., 0., 1.})).inverse(), 
        (Eigen::Translation3d(Eigen::Vector3d{1., 0., 0.}) * Eigen::AngleAxisd(M_PI / 4., Eigen::Vector3d{0., 0., 1.})).inverse(),
        (Eigen::Translation3d(Eigen::Vector3d{1., 1., 0.}) * Eigen::AngleAxisd(M_PI / 2., Eigen::Vector3d{0., 0., 1.})).inverse(),
        (Eigen::Translation3d(Eigen::Vector3d{0., 1., 0.}) * Eigen::AngleAxisd(-3 * M_PI / 4., Eigen::Vector3d{0., 0., 1.})).inverse()};
}

TEST(RotationAveraging, PerfectInitPerfectRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    const auto& view_pairs = CreateViewPairs(extrinsics);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics);
    
    theia::RobustRotationEstimator rotation_estimator(theia::RobustRotationEstimator::Options{});
    EXPECT_TRUE(rotation_estimator.EstimateRotations(view_pairs, &estimated_rotations));
    
    ExpectRotationsEqual(extrinsics, estimated_rotations, 1e-6);
}

TEST(RotationAveraging, PerfectInitOutlierRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    auto view_pairs = CreateViewPairs(extrinsics);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics);
    
    // add an outlier relative rotation
    auto& outlier_rot = view_pairs[theia::ViewIdPair{1, 2}].rotation_2;
    outlier_rot = -0.5 * outlier_rot;

    theia::RobustRotationEstimator rotation_estimator(theia::RobustRotationEstimator::Options{});
    EXPECT_TRUE(rotation_estimator.EstimateRotations(view_pairs, &estimated_rotations));
    
    ExpectRotationsEqual(extrinsics, estimated_rotations, 1e-4);
}

TEST(RotationAveraging, PerfectInitNoisyRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    // add noise to relative transformations
    const auto& view_pairs = CreateViewPairs(extrinsics, 0.05);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics);

    theia::RobustRotationEstimator rotation_estimator(theia::RobustRotationEstimator::Options{});
    EXPECT_TRUE(rotation_estimator.EstimateRotations(view_pairs, &estimated_rotations));
    
    ExpectRotationsEqual(extrinsics, estimated_rotations, 4e-2);
}

TEST(RotationAveraging, OutlierInitPerfectRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    const auto& view_pairs = CreateViewPairs(extrinsics);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics);
    // add outlier to the init poses
    estimated_rotations[2] = -0.5 * estimated_rotations[2];
    
    theia::RobustRotationEstimator rotation_estimator(theia::RobustRotationEstimator::Options{});
    EXPECT_TRUE(rotation_estimator.EstimateRotations(view_pairs, &estimated_rotations));
    
    ExpectRotationsEqual(extrinsics, estimated_rotations, 1e-6);
}

TEST(RotationAveraging, NoisyInitPerfectRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    const auto& view_pairs = CreateViewPairs(extrinsics);
    // add noise to init poses
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics, 0.03);
    
    theia::RobustRotationEstimator rotation_estimator(theia::RobustRotationEstimator::Options{});
    EXPECT_TRUE(rotation_estimator.EstimateRotations(view_pairs, &estimated_rotations));
    
    ExpectRotationsEqual(extrinsics, estimated_rotations, 3e-2);
}

TEST(RotationAveraging, NoisyInitNoisyRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    // add noise to relative transformations
    const auto& view_pairs = CreateViewPairs(extrinsics, 0.02);
    // add noise to init poses
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics, 0.03);
    
    theia::RobustRotationEstimator rotation_estimator(theia::RobustRotationEstimator::Options{});
    EXPECT_TRUE(rotation_estimator.EstimateRotations(view_pairs, &estimated_rotations));
    
    ExpectRotationsEqual(extrinsics, estimated_rotations, 3e-2);
}


TEST(TranslationAveraging, PerfectInitPerfectRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    const auto& view_pairs = CreateViewPairs(extrinsics);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics);

    theia::LeastUnsquaredDeviationPositionEstimator position_estimator(theia::LeastUnsquaredDeviationPositionEstimator::Options{});
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_positions{}; // no init for translations
    EXPECT_TRUE(position_estimator.EstimatePositions(view_pairs, estimated_rotations, &estimated_positions));

    ExpectTranslationsEqual(extrinsics, estimated_positions, 1e-6);
}

TEST(TranslationAveraging, PerfectInitOutlierRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    auto view_pairs = CreateViewPairs(extrinsics);
    // add an outlier relative translation
    auto& outlier_trans = view_pairs[theia::ViewIdPair{1, 2}].position_2;
    outlier_trans = -0.5 * outlier_trans;
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics);

    theia::LeastUnsquaredDeviationPositionEstimator position_estimator(theia::LeastUnsquaredDeviationPositionEstimator::Options{});
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_positions{}; // no init for translations
    EXPECT_TRUE(position_estimator.EstimatePositions(view_pairs, estimated_rotations, &estimated_positions));

    ExpectTranslationsEqual(extrinsics, estimated_positions, 1e-4);
}

TEST(TranslationAveraging, PerfectInitNoisyRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    // add noise to relative transformations
    const auto& view_pairs = CreateViewPairs(extrinsics, 0.05);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics);

    theia::LeastUnsquaredDeviationPositionEstimator position_estimator(theia::LeastUnsquaredDeviationPositionEstimator::Options{});
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_positions{}; // no init for translations
    EXPECT_TRUE(position_estimator.EstimatePositions(view_pairs, estimated_rotations, &estimated_positions));

    ExpectTranslationsEqual(extrinsics, estimated_positions, 5e-2);
}

TEST(TranslationAveraging, OutlierInitPerfectRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    const auto& view_pairs = CreateViewPairs(extrinsics);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics);
    // add outlier to the init poses
    estimated_rotations[1] = 0.9 * estimated_rotations[1];

    theia::LeastUnsquaredDeviationPositionEstimator position_estimator(theia::LeastUnsquaredDeviationPositionEstimator::Options{});
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_positions{}; // no init for translations
    EXPECT_TRUE(position_estimator.EstimatePositions(view_pairs, estimated_rotations, &estimated_positions));

    ExpectTranslationsEqual(extrinsics, estimated_positions, 1e-1);
}

TEST(TranslationAveraging, NoisyInitPerfectRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    const auto& view_pairs = CreateViewPairs(extrinsics);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics, 0.03);

    theia::LeastUnsquaredDeviationPositionEstimator position_estimator(theia::LeastUnsquaredDeviationPositionEstimator::Options{});
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_positions{}; // no init for translations
    EXPECT_TRUE(position_estimator.EstimatePositions(view_pairs, estimated_rotations, &estimated_positions));

    ExpectTranslationsEqual(extrinsics, estimated_positions, 4e-2);
}

TEST(TranslationAveraging, NoisyInitNoisyRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    const auto& view_pairs = CreateViewPairs(extrinsics, 0.03);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics, 0.03);

    theia::LeastUnsquaredDeviationPositionEstimator position_estimator(theia::LeastUnsquaredDeviationPositionEstimator::Options{});
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_positions{}; // no init for translations
    EXPECT_TRUE(position_estimator.EstimatePositions(view_pairs, estimated_rotations, &estimated_positions));

    ExpectTranslationsEqual(extrinsics, estimated_positions, 3e-2);
}

TEST(TransformationAveraging, NoisyInitNoisyRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    const auto& view_pairs = CreateViewPairs(extrinsics, 0.02);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics, 0.03);

    theia::RobustRotationEstimator rotation_estimator(theia::RobustRotationEstimator::Options{});
    EXPECT_TRUE(rotation_estimator.EstimateRotations(view_pairs, &estimated_rotations));
    ExpectRotationsEqual(extrinsics, estimated_rotations, 3e-2);

    theia::LeastUnsquaredDeviationPositionEstimator position_estimator(theia::LeastUnsquaredDeviationPositionEstimator::Options{});
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_positions{}; // no init for translations
    EXPECT_TRUE(position_estimator.EstimatePositions(view_pairs, estimated_rotations, &estimated_positions));
    ExpectTranslationsEqual(extrinsics, estimated_positions, 3e-2);
}

void WriteFile(const std::unordered_map<theia::ViewIdPair, theia::TwoViewInfo>& view_pairs, 
    const std::unordered_map<theia::ViewId, Eigen::Vector3d>& estimated_rotations, int n_views, const std::string& file)
{
    std::ofstream f(file);
    for (int id{0}; id != n_views; ++id)
    {
        const auto& est_rot = estimated_rotations.at(id);
        const auto& R = Eigen::AngleAxisd(est_rot.norm(), est_rot.normalized()).toRotationMatrix();
        f << std::setprecision(12) << id << "," << R(0, 0) << "," << R(1, 0) << "," << R(2, 0) << "," << R(0, 1) << "," << R(1, 1) << "," 
            << R(2, 1) << "," << R(0, 2) << "," << R(1, 2) << "," << R(2, 2) << "\n";
    }
    for (int id1{0}; id1 != n_views; ++id1)
    {
        for (int id0{0}; id0 != id1; ++id0)
        {
            const auto& two_view_info = view_pairs.at(theia::ViewIdPair{id0, id1});
            const auto& rot_vec = two_view_info.rotation_2;
            const auto& t = two_view_info.position_2;
            const auto& R = Eigen::AngleAxisd(rot_vec.norm(), rot_vec.normalized()).toRotationMatrix();
            f << std::setprecision(12) << id0 << "," << id1 << "," << R(0, 0) << "," << R(1, 0) << "," << R(2, 0) << "," << R(0, 1) << "," << R(1, 1) << "," 
                << R(2, 1) << "," << R(0, 2) << "," << R(1, 2) << "," << R(2, 2) << "," << t[0] << "," << t[1] << "," << t[2] << "\n";
        }
    }
}

TEST(BaInit, PerfectInitPerfectRel)
{
    const auto& extrinsics = CreateCameraExtrinsics();
    const auto& view_pairs = CreateViewPairs(extrinsics);
    std::unordered_map<theia::ViewId, Eigen::Vector3d> estimated_rotations = GetGlobalRotations(extrinsics);

    const std::string in_file("ba_init_in.csv");
    const std::string out_file("ba_init_out.csv");
    WriteFile(view_pairs, estimated_rotations, extrinsics.size(), in_file);

    BaInit ba_init(in_file);

    ba_init.Run();

    ba_init.WriteResult(out_file);
    std::cout << "Compare results in " << out_file << " to the target extrinsics: \n";
    for (const auto& extr : extrinsics)
    {
        std::cout << extr.matrix() << "\n" << "\n";
    }
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}