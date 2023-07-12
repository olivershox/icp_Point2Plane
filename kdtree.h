#ifndef KDTREE_H
#define KDTREE_H
#include "eigen_types.h"
#include "point_types.h"
#include <glog/logging.h>
#include <map>
#include <queue>
#include <unordered_map>

struct kdTreeNode {
    int id_ = -1;
    int point_idx_ =0;
    int axis_index_ = 0;
    float split_thresh_ = 0.0;
    kdTreeNode* left_ = nullptr;
    kdTreeNode* right_ = nullptr;
    
    bool isLeaf( ) const { return left_ == nullptr && right_ == nullptr;}

};

struct NodeAndDistance {
    NodeAndDistance(kdTreeNode* node , float dis2) : node_(node),distance2_(dis2){}
    kdTreeNode* node_ = nullptr;
    float distance2_ = 0;
    bool operator<(const NodeAndDistance& other) const {return distance2_<other.distance2_;}
};

class KdTree{
    public:
        explicit KdTree() = default;
        ~KdTree() {Clear();}
        bool BuildTree (const CloudPtr& cloud);
        bool GetClosestPoint(const PointType& pt , std::vector<int>& closest_idx , int k = 5);
        bool GetClosestPointMT(const CloudPtr& cloud, std::vector<std::pair<size_t,size_t>>&matches, int k = 5);//并行
        void SetEnableANN(bool use_ann = true ,float alpha = 0.1){
            approximate_ = use_ann;
            alpha_ =  alpha;
        }
        size_t size() const{ return size_;}
        void Clear();
        void PrintALL();
    private:
        void Insert(const IndexVec& points, kdTreeNode* node);
        bool FindSplitAxisAndThresh(const IndexVec& pint_idx, int& axis,float& th , IndexVec& left ,IndexVec& right);
        void Reset();
        static inline float Dis2(const Vec3f& p1 ,const Vec3f& p2) {return (p1-p2).squaredNorm();}
        void Knn(const Vec3f& pt , kdTreeNode* node ,std::priority_queue<NodeAndDistance>& result) const;
        void ComputeDisForLeaf (const Vec3f& pt ,kdTreeNode* node ,std::priority_queue<NodeAndDistance>& result) const;
        bool NeedExpand(const Vec3f& pt ,kdTreeNode* node ,std::priority_queue<NodeAndDistance>& knn_result) const;
        int k_ =5; //knn最近临个数
        std::shared_ptr<kdTreeNode> root_ = nullptr;
        std::vector<Vec3f> cloud_;
        std::unordered_map<int ,kdTreeNode*> nodes_;
        size_t size_=0;//叶子节点
        int tree_node_id_=0;
        bool approximate_= true;
        float alpha_ =0.1;
};
#endif


        
        
