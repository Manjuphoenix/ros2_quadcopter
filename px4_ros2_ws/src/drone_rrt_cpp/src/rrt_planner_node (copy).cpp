// rrt_planner_node.cpp
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

#include <rclcpp/qos.hpp>


namespace ob = ompl::base;
namespace og = ompl::geometric;


class RRTPlannerNode : public rclcpp::Node {
public:
  RRTPlannerNode() : Node("rrt_planner_node") {
    rclcpp::QoS qos(rclcpp::KeepLast(1));
    qos.transient_local();

    octomap_sub_ = create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_binary", qos,
      std::bind(&RRTPlannerNode::octomapCallback, this, std::placeholders::_1));
    rclcpp::QoS path_qos(rclcpp::KeepLast(1));
    path_qos.transient_local();
    path_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/rrt_path", path_qos);
    

    RCLCPP_INFO(get_logger(), "RRT planner node started, waiting for /octomap_binary...");
  }

private:
  std::shared_ptr<octomap::OcTree> tree_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr path_pub_;

  void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg) {
    octomap::AbstractOcTree* abstract = octomap_msgs::binaryMsgToMap(*msg);
    tree_.reset(dynamic_cast<octomap::OcTree*>(abstract));
    if (tree_) planPath();
  }

  bool isStateValid(const ob::State *state) {
    const auto *pos = state->as<ob::RealVectorStateSpace::StateType>();
    double x = pos->values[0], y = pos->values[1], z = pos->values[2];
    octomap::OcTreeNode* node = tree_->search(x, y, z);
    if (!node) return true; // unknown space treated as free (or set false to be conservative)
    return !tree_->isNodeOccupied(node);
  }

  void planPath() {
    auto space = std::make_shared<ob::RealVectorStateSpace>(3);
    ob::RealVectorBounds bounds(3);
    bounds.setLow(0, -20); bounds.setHigh(0, 20);
    bounds.setLow(1, -20); bounds.setHigh(1, 20);
    bounds.setLow(2, 0);   bounds.setHigh(2, 5);
    space->setBounds(bounds);

    og::SimpleSetup ss(space);
    ss.setStateValidityChecker(std::bind(&RRTPlannerNode::isStateValid, this, std::placeholders::_1));

    ob::ScopedState<> start(space), goal(space);
    start[0]=0; start[1]=0; start[2]=1.0;
    goal[0]=80; goal[1]=10; goal[2]=4;
    // goal[0]=15; goal[1]=5; goal[2]=1.5;
    ss.setStartAndGoalStates(start, goal);

    auto planner = std::make_shared<og::RRTstar>(ss.getSpaceInformation());
    ss.setPlanner(planner);

    if (ss.solve(5.0)) {
      ss.simplifySolution();
      auto &path = ss.getSolutionPath();
      geometry_msgs::msg::PoseArray path_msg;
      path_msg.header.frame_id = "map";
      for (size_t i = 0; i < path.getStateCount(); ++i) {
        const auto *s = path.getState(i)->as<ob::RealVectorStateSpace::StateType>();
        geometry_msgs::msg::Pose p;
        p.position.x = s->values[0];
        p.position.y = s->values[1];
        p.position.z = s->values[2];
        path_msg.poses.push_back(p);
      }
      path_pub_->publish(path_msg);
      RCLCPP_INFO(get_logger(), "RRT* path found with %zu waypoints", path.getStateCount());
    } else {
      RCLCPP_WARN(get_logger(), "No solution found");
    }
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RRTPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
