#ifndef ICP_3D_H
#define ICP_3D_H
#include "kdtree.h"
class Icp3d{
   public:
    struct Options{
        int max_iteration = 20;
        double max_plane_distance_ = 0.05;
        int min_effective_pts_ = 10;
        double eps_ = 1e-2;
        bool use_initial_translation_ =false;
    };

    Icp3d() {}
    Icp3d(Options options) : options_(options) {}
    void SetTarget(CloudPtr target) {
    target_ = target;
    BuildTargetKdTree();

    // 计算点云中心
    target_center_ = std::accumulate(target->points.begin(), target_->points.end(), Vec3d::Zero().eval(),
                                        [](const Vec3d& c, const PointType& pt) -> Vec3d { return c + ToVec3d(pt); }) /
                        target_->size();
    LOG(INFO) << "target center: " << target_center_.transpose();
}
    /// 设置被配准的Scan
    void SetSource(CloudPtr source) {
        source_ = source;
        source_center_ = std::accumulate(source_->points.begin(), source_->points.end(), Vec3d::Zero().eval(),
                                         [](const Vec3d& c, const PointType& pt) -> Vec3d { return c + ToVec3d(pt); }) /
                         source_->size();
        LOG(INFO) << "source center: " << source_center_.transpose();
    }
    bool AlignP2Plane(SE3& init_pose);
    
   private:
    void BuildTargetKdTree();
    std::shared_ptr<KdTree> kdtree_ = nullptr;
    CloudPtr target_ = nullptr;
    CloudPtr source_ = nullptr;
    Vec3d target_center_ = Vec3d::Zero();
    Vec3d source_center_ = Vec3d::Zero();
    Options options_;
};
#endif
