#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "System.h"

using namespace std;
using namespace ORB_SLAM3;

class StereoSlamNode : public rclcpp::Node
{
public:
  StereoSlamNode(const string &vocab_path, const string &settings_path)
  : Node("stereo_slam_node")
  {
    // 初始化 ORB-SLAM3 双目
    slam_ = new System(vocab_path, settings_path, System::STEREO, true, false);

    // 订阅左右图
    left_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/left/image_raw", 10,
      [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
        left_cb(msg);
      });

    right_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/right/image_raw", 10,
      [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
        right_cb(msg);
      });

    RCLCPP_INFO(this->get_logger(), "Stereo SLAM node started, waiting for images...");
  }

  ~StereoSlamNode()
  {
    slam_->Shutdown();
    delete slam_;
  }

private:
  void left_cb(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    try {
      left_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
      left_stamp_ = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    } catch (...) {
      return;
    }
    Process();
  }

  void right_cb(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    try {
      right_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
      right_stamp_ = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    } catch (...) {
      return;
    }
    Process();
  }

  void Process()
  {
    if (left_.empty() || right_.empty()) return;
    // 简单对齐：用左图时间戳
    slam_->TrackStereo(left_, right_, left_stamp_);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_sub_, right_sub_;
  cv::Mat left_, right_;
  double left_stamp_, right_stamp_;
  System *slam_;
};

int main(int argc, char **argv)
{
  if (argc != 3) {
    cerr << "Usage: ros2 run your_pkg stereo_node VOCAB_PATH SETTINGS_PATH" << endl;
    return 1;
  }

  rclcpp::init(argc, argv);
  auto node = make_shared<StereoSlamNode>(argv[1], argv[2]);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}