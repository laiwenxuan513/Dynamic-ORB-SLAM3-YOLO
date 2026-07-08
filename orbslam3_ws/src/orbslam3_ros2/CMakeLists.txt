cmake_minimum_required(VERSION 3.8)
project(orbslam3_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

set(ORB_SLAM3_ROOT /home/elf/ORB_SLAM3)

include_directories(
  ${ORB_SLAM3_ROOT}/include
  ${ORB_SLAM3_ROOT}/include/CameraModels
  ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
  ${ORB_SLAM3_ROOT}/Thirdparty/g2o
  ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2
)

link_directories(
  ${ORB_SLAM3_ROOT}/lib
  ${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib
  ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib
)

add_executable(stereo_node stereo_node.cpp)
ament_target_dependencies(stereo_node rclcpp sensor_msgs cv_bridge)
target_link_libraries(stereo_node ORB_SLAM3 ${OpenCV_LIBS} g2o DBoW2 pthread)

install(TARGETS stereo_node DESTINATION lib/${PROJECT_NAME})
ament_package()
