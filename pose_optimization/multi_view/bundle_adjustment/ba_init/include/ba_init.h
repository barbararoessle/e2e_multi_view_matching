#ifndef BA_INIT_H
#define BA_INIT_H

#include <iostream>
#include <unordered_map>

#include <Eigen/Core>
#include <theia/theia.h>

class BaInit
{
  public:
    BaInit(const std::string& file_path);
    ~BaInit();
    void Run();
    void WriteResult(const std::string& file_path) const;

  private:
    int n_views_{0};
    std::unordered_map<theia::ViewIdPair, theia::TwoViewInfo> view_pairs_{};
    std::unordered_map<theia::ViewId, Eigen::Vector3d> global_rotations_{};
    std::unordered_map<theia::ViewId, Eigen::Vector3d> global_translations_{};
};

#endif // BA_INIT_H