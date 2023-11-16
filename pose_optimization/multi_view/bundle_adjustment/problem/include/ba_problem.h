#ifndef BA_PROBLEM_H
#define BA_PROBLEM_H

#include <ceres/ceres.h>
#include <ceres/rotation.h>

class BaProblem {
  public:
    BaProblem(const std::string& file_path);
    ~BaProblem();

    int NumberDimensions() const { return n_dims_; }
    int num_observations() const { return n_obs_; }
    const double* observations() const { return observations_; }
    const double* observation_weights() const { return observation_weights_; }
    double* mutable_cameras() { return parameters_; }
    double* mutable_points() { return parameters_ + 6 * n_cams_; }

    bool is_observed_by_fixed_camera(int i)
    {
        return camera_index_[i] == fixed_cam_id_;
    }
    double* mutable_camera_for_observation(int i)
    {
        return mutable_cameras() + camera_index_[i] * 6;
    }
    double* mutable_point_for_observation(int i)
    {
        return mutable_points() + point_index_[i] * 3;
    }
    double GetFx() const { return f_x_; }
    double GetFy() const { return f_y_; }
    double GetCx() const { return c_x_; }
    double GetCy() const { return c_y_; }

    void WriteResult(const std::string& file_path) const;

  private:
    int n_dims_; // dimension of an observation
    int n_cams_; // number of cameras
    int fixed_cam_id_; // index of camera pose that is fixed during optimization
    int n_pts_; // number of 3D points
    int n_obs_; // number of observations
    int n_params_; // number of parameters

    // data
    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* observation_weights_;
    double* parameters_;

    // intrinsics
    double f_x_;
    double f_y_;
    double c_x_;
    double c_y_;
};

struct ReprojectionErrorFixedCamera {
  ReprojectionErrorFixedCamera(double observed_x, double observed_y, double f_x, double f_y, double c_x, double c_y, double w_x, double w_y)
      : observed_x_(observed_x), observed_y_(observed_y), f_x_(f_x), f_y_(f_y), c_x_(c_x), c_y_(c_y), w_x_(w_x), w_y_(w_y) {}

  template <typename T>
  bool operator()(const T* const point,
                  T* residuals) const
  {
    T predicted_x = f_x_ * point[0] / point[2] + c_x_;
    T predicted_y = f_y_ * point[1] / point[2] + c_y_;

    // residual is the difference between the predicted and observed position.
    residuals[0] = w_x_ * (predicted_x - observed_x_);
    residuals[1] = w_y_ * (predicted_y - observed_y_);

    return true;
  }

  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y, 
                                     const double f_x, 
                                     const double f_y, 
                                     const double c_x, 
                                     const double c_y,
                                     const double w_x,
                                     const double w_y)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorFixedCamera, 2, 3>(
            new ReprojectionErrorFixedCamera(observed_x, observed_y, f_x, f_y, c_x, c_y, w_x, w_y)));
    }

    double observed_x_;
    double observed_y_;

    double f_x_;
    double f_y_;
    double c_x_;
    double c_y_;
    double w_x_;
    double w_y_;
};

struct ReprojectionError {
  ReprojectionError(double observed_x, double observed_y, double f_x, double f_y, double c_x, double c_y, double w_x, double w_y)
      : observed_x_(observed_x), observed_y_(observed_y), f_x_(f_x), f_y_(f_y), c_x_(c_x), c_y_(c_y), w_x_(w_x), w_y_(w_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const
{
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    T predicted_x = f_x_ * p[0] / p[2] + c_x_;
    T predicted_y = f_y_ * p[1] / p[2] + c_y_;

    // residual is the difference between the predicted and observed position.
    residuals[0] = w_x_ * (predicted_x - observed_x_);
    residuals[1] = w_y_ * (predicted_y - observed_y_);

    return true;
}

  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y, 
                                     const double f_x, 
                                     const double f_y, 
                                     const double c_x, 
                                     const double c_y,
                                     const double w_x,
                                     const double w_y)
    {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(observed_x, observed_y, f_x, f_y, c_x, c_y, w_x, w_y)));
    }

    double observed_x_;
    double observed_y_;

    double f_x_;
    double f_y_;
    double c_x_;
    double c_y_;
    double w_x_;
    double w_y_;
};

void Solve(BaProblem& ba_problem);

#endif // BA_PROBLEM_H