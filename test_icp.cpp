#include <sophus/se3.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <icp_3d.h>
DEFINE_string(source, "./data/1666079301.105095.pcd", "第1个点云路径");
DEFINE_string(target, "./data/1666079301.204635.pcd", "第2个点云路径");
using namespace std;
using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;
using PointVec = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
using IndexVec = std::vector<int>;
using SE3f = Sophus::SE3f;
using SE3 = Sophus::SE3d;
using Quatd = Eigen::Quaterniond;
using Vec3d = Eigen::Vector3d;
template<typename CloudType> 
void SaveCloudToFile(const std::string &filePath, CloudType &cloud) {
    cloud.height = 1;
    cloud.width = cloud.size();
    pcl::io::savePCDFileASCII(filePath, cloud);
}
int main(int argc ,char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);
    CloudPtr source(new PointCloudType), target(new PointCloudType);
    cout<<fLS::FLAGS_source<<endl;
    cout<<fLS::FLAGS_target<<endl;
    pcl::io::loadPCDFile(fLS::FLAGS_source, *source);
    pcl::io::loadPCDFile(fLS::FLAGS_target, *target);

    Icp3d icp;
    bool success;
    icp.SetSource(source);
    icp.SetTarget(target);
    SE3 pose;
    success = icp.AlignP2Plane(pose);
    if (success){
        CloudPtr source_trans(new PointCloudType);
        pcl::transformPointCloud(*source, *source_trans, pose.matrix().cast<float>());
        CloudPtr output_in(new PointCloudType);
        *output_in = *source_trans+*target;
        LOG(INFO) << "icp p2plane align success, pose: " << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << pose.translation().transpose();
        SaveCloudToFile("./data/pcl_icp_trans.pcd", *output_in);
    }
    else{
        LOG(ERROR) << "align failed.";
    }
    return 0;
    

}
