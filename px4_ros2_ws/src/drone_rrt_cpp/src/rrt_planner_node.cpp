// rrt_planner_node.cpp
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>

namespace ob = ompl::base;
namespace og = ompl::geometric;

class RRTPlannerNode : public rclcpp::Node {
public:
  RRTPlannerNode() : Node("rrt_planner_node") {
    // Declare goal as parameters so it can be set at launch without recompiling
    this->declare_parameter("goal_x", 15.0);
    this->declare_parameter("goal_y", 5.0);
    this->declare_parameter("goal_z", 1.5);

    // Bounds are also parameterized in case your world extends beyond +/-20
    this->declare_parameter("bound_xy", 20.0);
    this->declare_parameter("bound_z_max", 5.0);

    // QoS matching octomap_server's transient_local ("latched") publisher
    rclcpp::QoS map_qos(rclcpp::KeepLast(1));
    map_qos.transient_local();

    octomap_sub_ = create_subscription<octomap_msgs::msg::Octomap>(
      "/octomap_binary", map_qos,
      std::bind(&RRTPlannerNode::octomapCallback, this, std::placeholders::_1));

    // QoS matching PX4's uXRCE-DDS agent output topics (best effort, volatile)
    rclcpp::QoS px4_qos(rclcpp::KeepLast(5));
    px4_qos.best_effort();
    px4_qos.durability_volatile();

    position_sub_ = create_subscription<px4_msgs::msg::VehicleLocalPosition>(
      "/fmu/out/vehicle_local_position_v1", px4_qos,
      std::bind(&RRTPlannerNode::positionCallback, this, std::placeholders::_1));

    // Path publisher: transient_local so a late-starting waypoint follower still gets it
    rclcpp::QoS path_qos(rclcpp::KeepLast(1));
    // path_qos.transient_local();
    
    
    
    path_qos.reliable();
    path_qos.durability_volatile();
    path_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/rrt_path", path_qos);
    
    
    path_publish_timer_ = create_wall_timer(
        std::chrono::milliseconds(200),
        std::bind(&RRTPlannerNode::republishPath, this));

    RCLCPP_INFO(get_logger(), "RRT planner node started.");
    RCLCPP_INFO(get_logger(), "Waiting for /octomap_binary and /fmu/out/vehicle_local_position_v1 ...");
  }

private:
  std::shared_ptr<octomap::OcTree> tree_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr position_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr path_pub_;
  

  bool have_map_ = false;
  bool have_position_ = false;
  bool planned_ = false;  // plan once; set false again if you want continuous replanning

  double start_x_ = 0.0, start_y_ = 0.0, start_z_ = 0.0;
  
  
  
  geometry_msgs::msg::PoseArray last_path_msg_;
  bool have_path_ = false;
  rclcpp::TimerBase::SharedPtr path_publish_timer_;
  
  rclcpp::Time last_attempt_;
  
  

  void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg) {
    octomap::AbstractOcTree* abstract = octomap_msgs::binaryMsgToMap(*msg);
    tree_.reset(dynamic_cast<octomap::OcTree*>(abstract));
    if (tree_) {
      if (!have_map_) {
        RCLCPP_INFO(get_logger(), "Octomap received (%zu leaf nodes).", tree_->calcNumNodes());
      }
      have_map_ = true;
      tryPlan();
    } else {
      RCLCPP_WARN(get_logger(), "Failed to convert message to OcTree");
    }
  }

  void positionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg) {
    if (!msg->xy_valid || !msg->z_valid) {
      return;  // don't trust the estimate until PX4 says it's valid
    }
    // PX4 local position is NED (x=North, y=East, z=Down).
    // Convert to the planner's ENU-style, Z-up convention: negate Z.
    start_x_ = msg->x;
    start_y_ = msg->y;
    start_z_ = -msg->z;

    if (!have_position_) {
      RCLCPP_INFO(get_logger(), "Vehicle position received: (%.2f, %.2f, %.2f) [ENU]",
                  start_x_, start_y_, start_z_);
    }
    have_position_ = true;
    tryPlan();
  }
  

  void tryPlan() {
    if (have_map_ && have_position_ && !planned_) {
      auto now = this->get_clock()->now();
      if (last_attempt_.nanoseconds() != 0 &&
          (now - last_attempt_).seconds() < 2.0){
          return;  // wait atleast 2 seconds between attempts
          }
      last_attempt_ = now;
      planned_ = true;  // remove this line (and re-arm it elsewhere) if you want repeated replanning
      planPath();
    }
  }
  
  void republishPath() {
      if(have_path_){
      last_path_msg_.header.stamp = this->get_clock()->now();
      path_pub_->publish(last_path_msg_);
      }
  }

  bool isStateValid(const ob::State *state) {
    const auto *pos = state->as<ob::RealVectorStateSpace::StateType>();
    double x = pos->values[0], y = pos->values[1], z = pos->values[2];
    octomap::OcTreeNode* node = tree_->search(x, y, z);
    if (!node) return true;  // unknown space treated as free
    return !tree_->isNodeOccupied(node);
  }

  void planPath() {
    double bound_xy = this->get_parameter("bound_xy").as_double();
    double bound_z_max = this->get_parameter("bound_z_max").as_double();
    double goal_x = this->get_parameter("goal_x").as_double();
    double goal_y = this->get_parameter("goal_y").as_double();
    double goal_z = this->get_parameter("goal_z").as_double();
    

    auto space = std::make_shared<ob::RealVectorStateSpace>(3);
    ob::RealVectorBounds bounds(3);
    bounds.setLow(0, -bound_xy);  bounds.setHigh(0, bound_xy);
    bounds.setLow(1, -bound_xy);  bounds.setHigh(1, bound_xy);
    bounds.setLow(2, -0.5);        bounds.setHigh(2, bound_z_max);
    space->setBounds(bounds);

    og::SimpleSetup ss(space);
    ss.setStateValidityChecker(std::bind(&RRTPlannerNode::isStateValid, this, std::placeholders::_1));

    ob::ScopedState<> start(space), goal(space);
    start[0] = start_x_; start[1] = start_y_; start[2] = start_z_;
    goal[0] = goal_x;    goal[1] = goal_y;    goal[2] = goal_z;

    RCLCPP_INFO(get_logger(), "Planning from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)",
                start_x_, start_y_, start_z_, goal_x, goal_y, goal_z);

    ss.setStartAndGoalStates(start, goal);

    auto planner = std::make_shared<og::RRTstar>(ss.getSpaceInformation());
    ss.setPlanner(planner);

    if (ss.solve(5.0)) {
      ss.simplifySolution();
      auto &path = ss.getSolutionPath();
      geometry_msgs::msg::PoseArray path_msg;
      path_msg.header.frame_id = "map";
      path_msg.header.stamp = this->get_clock()->now();
      for (size_t i = 0; i < path.getStateCount(); ++i) {
        const auto *s = path.getState(i)->as<ob::RealVectorStateSpace::StateType>();
        geometry_msgs::msg::Pose p;
        p.position.x = s->values[0];
        p.position.y = s->values[1];
        p.position.z = s->values[2];
        path_msg.poses.push_back(p);
      }
      last_path_msg_ = path_msg;
      have_path_ = true;
      path_pub_->publish(path_msg);
      RCLCPP_INFO(get_logger(), "RRT* path found with %zu waypoints", path.getStateCount());
    } else {
      RCLCPP_WARN(get_logger(), "No solution found");
      planned_ = false;  // allow retry on next map/position update
    }
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RRTPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
