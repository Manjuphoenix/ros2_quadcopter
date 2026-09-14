cmake_minimum_required(VERSION 3.8)
project(drone_rrt_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(px4_msgs REQUIRED)
pkg_check_modules(OMPL REQUIRED ompl)

add_executable(rrt_planner_node src/rrt_planner_node.cpp)

ament_target_dependencies(rrt_planner_node
  rclcpp
  octomap_msgs
  geometry_msgs
  px4_msgs
)

target_include_directories(rrt_planner_node PUBLIC
  ${OMPL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)
target_link_directories(rrt_planner_node PUBLIC
  ${OMPL_LIBRARY_DIRS}
)
target_link_libraries(rrt_planner_node
  ${OMPL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

install(TARGETS
  rrt_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
