// rrt_planner_node.cpp
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/PlannerData.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>

#include <gz/transport/Node.hh>
#include <gz/msgs/marker.pb.h>
#include <gz/msgs/Utility.hh>
#include <gz/msgs/empty.pb.h>      // ← add: reply type of the /marker service

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

    // Planner selection: "rrt" or "rrtstar"
    this->declare_parameter("planner", std::string("rrtstar"));
    
    // constructor
    this->declare_parameter("spawn_x", 0.0);
    this->declare_parameter("spawn_y", 0.0);
    
    // clearance
    this->declare_parameter("clearance", 0.5);

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


  // Tree: each consecutive pair of poses is one edge (parent, child).
  // transient_local = "latched": a recorder that starts later still gets the last tree.
  rclcpp::QoS tree_qos(rclcpp::KeepLast(1));
  tree_qos.reliable();
  tree_qos.transient_local();
  tree_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/rrt_tree", tree_qos);


    RCLCPP_INFO(get_logger(), "RRT planner node started.");
    RCLCPP_INFO(get_logger(), "Waiting for /octomap_binary and /fmu/out/vehicle_local_position_v1 ...");
  }

private:
  std::shared_ptr<octomap::OcTree> tree_;
  gz::transport::Node gz_node_;   // class member
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr position_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr path_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr tree_pub_;
  
  
  void drawGz(const std::string &ns, int id,
            gz::msgs::Marker::Type type,
            const std::vector<std::array<double,3>> &pts,
            double r, double g, double b) {
  gz::msgs::Marker m;
  m.set_ns(ns);
  m.set_id(id);
  m.set_action(gz::msgs::Marker::ADD_MODIFY);
  m.set_type(type);                       // LINE_LIST for tree, LINE_STRIP for path
  m.set_visibility(gz::msgs::Marker::GUI);
  for (auto *c : {m.mutable_material()->mutable_ambient(),
                  m.mutable_material()->mutable_diffuse(),
                  m.mutable_material()->mutable_emissive()}) {
    c->set_r(r); c->set_g(g); c->set_b(b); c->set_a(1.0);
  }
  for (auto &p : pts)
    gz::msgs::Set(m.add_point(), gz::math::Vector3d(p[0], p[1], p[2]));
  // gz_node_.Request("/marker", m);         // one-way request, doesn't block
  gz::msgs::Empty rep;
  bool ok = false;
  // Blocking call: waits up to 1000 ms until Gazebo has processed this marker
  if (!gz_node_.Request("/marker", m, 1000, rep, ok) || !ok)
    RCLCPP_WARN(get_logger(), "Marker request failed (ns=%s)", ns.c_str());
  }

  void clearGz(const std::string &ns) {
      gz::msgs::Marker m;
      m.set_ns(ns);
      m.set_action(gz::msgs::Marker::DELETE_ALL);

      gz::msgs::Empty rep;
      bool ok = false;
      if (!gz_node_.Request("/marker", m, 1000, rep, ok) || !ok)
        RCLCPP_WARN(get_logger(), "Marker clear failed (ns=%s)", ns.c_str());
  }
  

  bool have_map_ = false;
  double clearance_ = 0.5;   // metres; set from the "clearance" parameter in planPath()
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
    // start_x_ = msg->x;
    // start_y_ = msg->y;
    
    // start_x_ = msg->y;
    // start_y_ = msg->x;
    // start_z_ = -msg->z;
    
    // positionCallback
    start_x_ = msg->y + get_parameter("spawn_x").as_double();   // East  + world x offset
    start_y_ = msg->x + get_parameter("spawn_y").as_double();   // North + world y offset
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

  // bool isStateValid(const ob::State *state) {
  //     const auto *pos = state->as<ob::RealVectorStateSpace::StateType>();
  //     double x = pos->values[0], y = pos->values[1], z = pos->values[2];
  //     octomap::OcTreeNode* node = tree_->search(x, y, z);
  //     if (!node) return true;  // unknown space treated as free
  //     return !tree_->isNodeOccupied(node);
  //   }
  
  
  bool isStateValid(const ob::State *state) {
    const auto *pos = state->as<ob::RealVectorStateSpace::StateType>();
    const octomap::point3d p(pos->values[0], pos->values[1], pos->values[2]);
    const double r = clearance_;   // cache get_parameter("clearance") in planPath()

    const octomap::point3d lo(p.x() - r, p.y() - r, p.z() - r);
    const octomap::point3d hi(p.x() + r, p.y() + r, p.z() + r);
    for (auto it = tree_->begin_leafs_bbx(lo, hi), end = tree_->end_leafs_bbx(); it != end; ++it) {
      if (!tree_->isNodeOccupied(*it)) continue;
      // a pruned leaf can be bigger than one voxel, so allow for half its size
      if ((it.getCoordinate() - p).norm() <= r + it.getSize() * 0.5)
        return false;
    }
    return true;
  }

  void planPath() {
    double bound_xy = this->get_parameter("bound_xy").as_double();
    double bound_z_max = this->get_parameter("bound_z_max").as_double();
    double goal_x = this->get_parameter("goal_x").as_double();
    double goal_y = this->get_parameter("goal_y").as_double();
    double goal_z = this->get_parameter("goal_z").as_double();
    
    clearance_ = this->get_parameter("clearance").as_double();
    

    auto space = std::make_shared<ob::RealVectorStateSpace>(3);
    ob::RealVectorBounds bounds(3);
    bounds.setLow(0, -bound_xy);  bounds.setHigh(0, bound_xy);
    bounds.setLow(1, -bound_xy);  bounds.setHigh(1, bound_xy);
    bounds.setLow(2, -0.5);        bounds.setHigh(2, bound_z_max);
    space->setBounds(bounds);

    og::SimpleSetup ss(space);
    ss.setStateValidityChecker(std::bind(&RRTPlannerNode::isStateValid, this, std::placeholders::_1));
    
    const double extent = ss.getStateSpace()->getMaximumExtent();
    ss.getSpaceInformation()->setStateValidityCheckingResolution(0.1 / extent);

    ob::ScopedState<> start(space), goal(space);
    start[0] = start_x_; start[1] = start_y_; start[2] = start_z_;
    goal[0] = goal_x;    goal[1] = goal_y;    goal[2] = goal_z;

    RCLCPP_INFO(get_logger(), "Planning from (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)",
                start_x_, start_y_, start_z_, goal_x, goal_y, goal_z);

    ss.setStartAndGoalStates(start, goal);

    std::string planner_name = get_parameter("planner").as_string();
    ob::PlannerPtr planner;
    if (planner_name == "rrt")
      planner = std::make_shared<og::RRT>(ss.getSpaceInformation());
    else
      planner = std::make_shared<og::RRTstar>(ss.getSpaceInformation());
    ss.setPlanner(planner);
    RCLCPP_INFO(get_logger(), "Using planner: %s", planner->getName().c_str());

    if (ss.solve(0.5)) {
      ob::PlannerData pd(ss.getSpaceInformation());
      ss.getPlannerData(pd);           // every vertex and edge the planner kept
      
      std::vector<std::array<double,3>> tree_pts;   // pairs of points = edge segments
      
      for (unsigned int i = 0; i < pd.numVertices(); ++i) {
          std::vector<unsigned int> out;
          pd.getEdges(i, out);
          const auto *a = pd.getVertex(i).getState()->as<ob::RealVectorStateSpace::StateType>();
          for (unsigned int j : out) {
              const auto *b = pd.getVertex(j).getState()->as<ob::RealVectorStateSpace::StateType>();
              tree_pts.push_back({a->values[0], a->values[1], a->values[2]});
              tree_pts.push_back({b->values[0], b->values[1], b->values[2]});
          }
      }
      
      geometry_msgs::msg::PoseArray tree_msg;
      tree_msg.header.frame_id = "map";
      tree_msg.header.stamp = this->get_clock()->now();
      for (const auto &p : tree_pts) {             // already in world ENU, in edge pairs
        geometry_msgs::msg::Pose pose;
        pose.position.x = p[0];
        pose.position.y = p[1];
        pose.position.z = p[2];
        tree_msg.poses.push_back(pose);
      }
      tree_pub_->publish(tree_msg);


      // Draw the explored tree (grey) in Gazebo
      clearGz("rrt_tree");
      clearGz("rrt_path");
      // drawGz("rrt_tree", 0, gz::msgs::Marker::LINE_LIST, tree_pts, 0.6, 0.6, 0.6);
      drawGz("rrt_tree", 0, gz::msgs::Marker::LINE_LIST, tree_pts, 1.0, 1.0, 0.0);
      RCLCPP_INFO(get_logger(), "Tree: %u vertices, %zu edges sent to Gazebo",
                  pd.numVertices(), tree_pts.size() / 2);

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
      // Draw the final (simplified) path (green) in Gazebo
      std::vector<std::array<double,3>> path_pts;
      for (const auto &p : path_msg.poses)
        path_pts.push_back({p.position.x, p.position.y, p.position.z});
      drawGz("rrt_path", 1, gz::msgs::Marker::LINE_STRIP, path_pts, 0.0, 1.0, 0.0);

      last_path_msg_ = path_msg;
      have_path_ = true;
      path_pub_->publish(path_msg);
      RCLCPP_INFO(get_logger(), "%s path found with %zu waypoints",
                  planner_name.c_str(), path.getStateCount());
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
