cmake_minimum_required(VERSION 3.8)
project(dynamic_stereo_slam_camera)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

set(ORB_SLAM3_ROOT "/home/elf/elf/ORB_SLAM3")

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${ORB_SLAM3_ROOT}
  ${ORB_SLAM3_ROOT}/include
  ${ORB_SLAM3_ROOT}/include/CameraModels
  ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
)

add_executable(dynamic_stereo_slam_node
  src/dynamic_stereo_slam_node.cpp
)

ament_target_dependencies(dynamic_stereo_slam_node
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
)

target_link_libraries(dynamic_stereo_slam_node
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  ${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so
  ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so
)

install(TARGETS
  dynamic_stereo_slam_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

