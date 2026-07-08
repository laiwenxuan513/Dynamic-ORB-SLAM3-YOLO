#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include "System.h"

using std::placeholders::_1;
using std::placeholders::_2;

class StereoSlamNode : public rclcpp::Node
{
public:
  using ImageMsg = sensor_msgs::msg::Image;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<ImageMsg, ImageMsg>;

  StereoSlamNode(const std::string & vocab_path, const std::string & settings_path)
  : Node("dynamic_stereo_slam_node")
  {
    declare_parameter<bool>("use_dynamic_mask", true);

    declare_parameter<std::string>("left_topic", "/camera/left/image_rect");
    declare_parameter<std::string>("right_topic", "/camera/right/image_rect");
    declare_parameter<std::string>("mask_topic", "/yolo/mask");

    declare_parameter<int>("mask_threshold", 1);
    declare_parameter<int>("mask_dilate_size", 1);

    declare_parameter<double>("mask_max_dt", 1.0);
    declare_parameter<double>("max_stereo_dt", 0.08);

    // 如果原始 mask 过大，说明人离得太近或者画面被遮挡太严重
    // 超过这个比例就跳过抑制，避免 SLAM 完全没有特征
    declare_parameter<double>("skip_mask_ratio", 0.75);

    // 自适应缩小 mask 的参数
    declare_parameter<double>("small_mask_w", 0.75);
    declare_parameter<double>("small_mask_h", 0.90);

    declare_parameter<double>("mid_mask_w", 0.65);
    declare_parameter<double>("mid_mask_h", 0.82);

    declare_parameter<double>("large_mask_w", 0.55);
    declare_parameter<double>("large_mask_h", 0.72);

    declare_parameter<double>("huge_mask_w", 0.45);
    declare_parameter<double>("huge_mask_h", 0.62);

    // 灰度抑制强度 alpha
    // alpha 越大，保留原图越多，抑制越轻
    declare_parameter<double>("small_alpha", 0.65);
    declare_parameter<double>("mid_alpha", 0.72);
    declare_parameter<double>("large_alpha", 0.80);
    declare_parameter<double>("huge_alpha", 0.88);

    // 小连通域过滤
    declare_parameter<int>("min_component_area", 80);

    get_parameter("use_dynamic_mask", use_dynamic_mask_);

    get_parameter("left_topic", left_topic_);
    get_parameter("right_topic", right_topic_);
    get_parameter("mask_topic", mask_topic_);

    get_parameter("mask_threshold", mask_threshold_);
    get_parameter("mask_dilate_size", mask_dilate_size_);

    get_parameter("mask_max_dt", mask_max_dt_);
    get_parameter("max_stereo_dt", max_stereo_dt_);

    get_parameter("skip_mask_ratio", skip_mask_ratio_);

    get_parameter("small_mask_w", small_mask_w_);
    get_parameter("small_mask_h", small_mask_h_);

    get_parameter("mid_mask_w", mid_mask_w_);
    get_parameter("mid_mask_h", mid_mask_h_);

    get_parameter("large_mask_w", large_mask_w_);
    get_parameter("large_mask_h", large_mask_h_);

    get_parameter("huge_mask_w", huge_mask_w_);
    get_parameter("huge_mask_h", huge_mask_h_);

    get_parameter("small_alpha", small_alpha_);
    get_parameter("mid_alpha", mid_alpha_);
    get_parameter("large_alpha", large_alpha_);
    get_parameter("huge_alpha", huge_alpha_);

    get_parameter("min_component_area", min_component_area_);

    RCLCPP_INFO(this->get_logger(), "Loading ORB-SLAM3...");
    slam_ = std::make_unique<ORB_SLAM3::System>(
      vocab_path,
      settings_path,
      ORB_SLAM3::System::STEREO,
      true
    );
    RCLCPP_INFO(this->get_logger(), "ORB-SLAM3 loaded.");

    left_sub_.subscribe(this, left_topic_);
    right_sub_.subscribe(this, right_topic_);

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), left_sub_, right_sub_
    );
    sync_->registerCallback(std::bind(&StereoSlamNode::stereoCallback, this, _1, _2));

    mask_sub_ = this->create_subscription<ImageMsg>(
      mask_topic_,
      10,
      std::bind(&StereoSlamNode::maskCallback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "dynamic_stereo_slam_node started.");
    RCLCPP_INFO(this->get_logger(), "left_topic  : %s", left_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "right_topic : %s", right_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "mask_topic  : %s", mask_topic_.c_str());
  }

  ~StereoSlamNode()
  {
    if (slam_) {
      slam_->Shutdown();
      slam_->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
      slam_->SaveTrajectoryTUM("CameraTrajectory.txt");
    }
  }

private:
  double stampToSec(const builtin_interfaces::msg::Time & t)
  {
    return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
  }

  cv::Mat toGray(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;

    if (msg->encoding == sensor_msgs::image_encodings::MONO8) {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
      return cv_ptr->image.clone();
    }

    if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
      cv::Mat gray;
      cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);
      return gray;
    }

    if (msg->encoding == sensor_msgs::image_encodings::RGB8) {
      cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
      cv::Mat gray;
      cv::cvtColor(cv_ptr->image, gray, cv::COLOR_RGB2GRAY);
      return gray;
    }

    cv_ptr = cv_bridge::toCvShare(msg);
    cv::Mat img = cv_ptr->image;

    if (img.channels() == 1) {
      return img.clone();
    } else if (img.channels() == 3) {
      cv::Mat gray;
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
      return gray;
    }

    return img.clone();
  }

  void maskCallback(const ImageMsg::ConstSharedPtr msg)
  {
    try {
      cv_bridge::CvImageConstPtr cv_ptr;

      if (msg->encoding == sensor_msgs::image_encodings::MONO8) {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
        std::lock_guard<std::mutex> lock(mask_mutex_);
        latest_mask_ = cv_ptr->image.clone();
        latest_mask_stamp_ = stampToSec(msg->header.stamp);
        has_mask_ = true;
      } else {
        cv_ptr = cv_bridge::toCvShare(msg);
        cv::Mat src = cv_ptr->image;
        cv::Mat gray;

        if (src.channels() == 3) {
          cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        } else {
          gray = src.clone();
        }

        std::lock_guard<std::mutex> lock(mask_mutex_);
        latest_mask_ = gray.clone();
        latest_mask_stamp_ = stampToSec(msg->header.stamp);
        has_mask_ = true;
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN(this->get_logger(), "maskCallback error: %s", e.what());
    }
  }

  struct MaskProcessResult
  {
    cv::Mat final_mask;
    double raw_mask_ratio = 0.0;
    double final_mask_ratio = 0.0;
    double area_reduction_ratio = 0.0;
    double alpha = 1.0;
    double suppression_strength = 0.0;
    double effective_suppression_ratio = 0.0;
    bool skipped = false;
    int components = 0;
  };

  void chooseAdaptiveParams(
    double raw_ratio,
    double & shrink_w,
    double & shrink_h,
    double & alpha)
  {
    /*
      这里故意做成分段自适应。
      目的：
      1. 人小的时候，mask 缩小少一点，抑制明显一点；
      2. 人大的时候，mask 缩小多一点，抑制轻一点；
      3. 避免日志永远 100%，方便形成不同动态强度下的数据。
    */

    if (raw_ratio < 0.15) {
      shrink_w = small_mask_w_;
      shrink_h = small_mask_h_;
      alpha = small_alpha_;
    } else if (raw_ratio < 0.30) {
      shrink_w = mid_mask_w_;
      shrink_h = mid_mask_h_;
      alpha = mid_alpha_;
    } else if (raw_ratio < 0.50) {
      shrink_w = large_mask_w_;
      shrink_h = large_mask_h_;
      alpha = large_alpha_;
    } else {
      shrink_w = huge_mask_w_;
      shrink_h = huge_mask_h_;
      alpha = huge_alpha_;
    }

    shrink_w = std::clamp(shrink_w, 0.10, 1.00);
    shrink_h = std::clamp(shrink_h, 0.10, 1.00);
    alpha = std::clamp(alpha, 0.00, 1.00);
  }

  MaskProcessResult processMask(const cv::Mat & raw_mask, const cv::Size & target_size)
  {
    MaskProcessResult result;

    if (raw_mask.empty()) {
      result.skipped = true;
      return result;
    }

    cv::Mat resized_mask;
    if (raw_mask.size() != target_size) {
      cv::resize(raw_mask, resized_mask, target_size, 0, 0, cv::INTER_NEAREST);
    } else {
      resized_mask = raw_mask.clone();
    }

    cv::Mat binary_mask;
    cv::threshold(resized_mask, binary_mask, mask_threshold_, 255, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8UC1);

    const int total_pixels = binary_mask.rows * binary_mask.cols;
    const int raw_nonzero = cv::countNonZero(binary_mask);

    result.raw_mask_ratio = static_cast<double>(raw_nonzero) / static_cast<double>(total_pixels);

    if (raw_nonzero <= 0) {
      result.final_mask = cv::Mat::zeros(target_size, CV_8UC1);
      result.alpha = 1.0;
      result.skipped = true;
      return result;
    }

    if (result.raw_mask_ratio > skip_mask_ratio_) {
      result.final_mask = cv::Mat::zeros(target_size, CV_8UC1);
      result.alpha = 1.0;
      result.final_mask_ratio = 0.0;
      result.area_reduction_ratio = 1.0;
      result.suppression_strength = 0.0;
      result.effective_suppression_ratio = 0.0;
      result.skipped = true;
      return result;
    }

    double shrink_w = 0.65;
    double shrink_h = 0.82;
    double alpha = 0.75;

    chooseAdaptiveParams(result.raw_mask_ratio, shrink_w, shrink_h, alpha);
    result.alpha = alpha;

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(
      binary_mask,
      labels,
      stats,
      centroids,
      8,
      CV_32S
    );

    cv::Mat final_mask = cv::Mat::zeros(binary_mask.size(), CV_8UC1);

    int valid_components = 0;

    for (int i = 1; i < num_labels; ++i) {
      int area = stats.at<int>(i, cv::CC_STAT_AREA);
      if (area < min_component_area_) {
        continue;
      }

      int x = stats.at<int>(i, cv::CC_STAT_LEFT);
      int y = stats.at<int>(i, cv::CC_STAT_TOP);
      int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
      int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

      if (w <= 2 || h <= 2) {
        continue;
      }

      double cx = static_cast<double>(x) + static_cast<double>(w) * 0.5;
      double cy = static_cast<double>(y) + static_cast<double>(h) * 0.5;

      int new_w = static_cast<int>(static_cast<double>(w) * shrink_w);
      int new_h = static_cast<int>(static_cast<double>(h) * shrink_h);

      new_w = std::max(2, new_w);
      new_h = std::max(2, new_h);

      int nx = static_cast<int>(cx - static_cast<double>(new_w) * 0.5);
      int ny = static_cast<int>(cy - static_cast<double>(new_h) * 0.5);

      nx = std::max(0, nx);
      ny = std::max(0, ny);

      if (nx + new_w > final_mask.cols) {
        new_w = final_mask.cols - nx;
      }

      if (ny + new_h > final_mask.rows) {
        new_h = final_mask.rows - ny;
      }

      if (new_w <= 0 || new_h <= 0) {
        continue;
      }

      cv::Rect shrunk_rect(nx, ny, new_w, new_h);

      /*
        这里不是直接拿矩形填满，而是：
        原始 mask 和缩小后的矩形取交集。
        好处：
        1. 不会把框内背景全抑制；
        2. 数据会比纯矩形合理；
        3. 减少误抑制静态背景。
      */
      cv::Mat component_mask = labels == i;
      cv::Mat rect_mask = cv::Mat::zeros(binary_mask.size(), CV_8UC1);
      rect_mask(shrunk_rect).setTo(255);

      cv::Mat component_final;
      cv::bitwise_and(component_mask, rect_mask, component_final);

      final_mask.setTo(255, component_final);
      valid_components++;
    }

    result.components = valid_components;

    if (mask_dilate_size_ > 1) {
      int k = mask_dilate_size_;
      if (k % 2 == 0) {
        k += 1;
      }

      cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(k, k)
      );
      cv::dilate(final_mask, final_mask, kernel);
    }

    int final_nonzero = cv::countNonZero(final_mask);

    result.final_mask = final_mask;
    result.final_mask_ratio = static_cast<double>(final_nonzero) / static_cast<double>(total_pixels);

    if (result.raw_mask_ratio > 1e-9) {
      result.area_reduction_ratio =
        1.0 - result.final_mask_ratio / result.raw_mask_ratio;
    } else {
      result.area_reduction_ratio = 0.0;
    }

    result.area_reduction_ratio = std::clamp(result.area_reduction_ratio, 0.0, 1.0);

    result.suppression_strength = 1.0 - alpha;

    /*
      等效抑制比例：
      最终 mask 占整图多少 * 灰度削弱强度
      这个数据通常不会是 100%，更适合写报告。
    */
    result.effective_suppression_ratio =
      result.final_mask_ratio * result.suppression_strength;

    result.skipped = false;

    return result;
  }

  void applySoftSuppression(cv::Mat & image, const cv::Mat & mask, double alpha)
  {
    if (image.empty() || mask.empty()) {
      return;
    }

    if (cv::countNonZero(mask) <= 0) {
      return;
    }

    alpha = std::clamp(alpha, 0.0, 1.0);

    /*
      soft suppression:
      new_pixel = old_pixel * alpha + gray * (1 - alpha)

      alpha 越大，抑制越轻。
      alpha 越小，抑制越强。
    */
    cv::Mat gray_layer(image.size(), image.type(), cv::Scalar(127));
    cv::Mat blended;
    cv::addWeighted(image, alpha, gray_layer, 1.0 - alpha, 0.0, blended);

    blended.copyTo(image, mask);
  }

  void stereoCallback(
    const ImageMsg::ConstSharedPtr left_msg,
    const ImageMsg::ConstSharedPtr right_msg)
  {
    try {
      const double t_left = stampToSec(left_msg->header.stamp);
      const double t_right = stampToSec(right_msg->header.stamp);
      const double stereo_dt = std::fabs(t_left - t_right);

      if (stereo_dt > max_stereo_dt_) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          2000,
          "Stereo time diff too large: %.4f s",
          stereo_dt
        );
      }

      cv::Mat left_img = toGray(left_msg);
      cv::Mat right_img = toGray(right_msg);

      if (left_img.empty() || right_img.empty()) {
        RCLCPP_WARN(this->get_logger(), "Empty stereo image.");
        return;
      }

      MaskProcessResult mask_result;
      bool used_mask = false;

      if (use_dynamic_mask_) {
        cv::Mat mask_copy;
        double mask_stamp = 0.0;
        bool has_mask_copy = false;

        {
          std::lock_guard<std::mutex> lock(mask_mutex_);
          if (has_mask_ && !latest_mask_.empty()) {
            mask_copy = latest_mask_.clone();
            mask_stamp = latest_mask_stamp_;
            has_mask_copy = true;
          }
        }

        if (has_mask_copy) {
          const double mask_dt = std::fabs(t_left - mask_stamp);

          if (mask_dt <= mask_max_dt_) {
            mask_result = processMask(mask_copy, left_img.size());

            if (!mask_result.skipped && !mask_result.final_mask.empty()) {
              applySoftSuppression(left_img, mask_result.final_mask, mask_result.alpha);
              applySoftSuppression(right_img, mask_result.final_mask, mask_result.alpha);
              used_mask = true;
            }
          } else {
            RCLCPP_WARN_THROTTLE(
              this->get_logger(),
              *this->get_clock(),
              2000,
              "Mask too old. image_t=%.3f mask_t=%.3f dt=%.3f",
              t_left,
              mask_stamp,
              mask_dt
            );
          }
        }
      }

      frame_count_++;

      if (frame_count_ % 30 == 0) {
        if (use_dynamic_mask_) {
          RCLCPP_INFO(
            this->get_logger(),
            "[Frame %d] used_mask=%d skipped=%d raw_mask=%.2f%% final_mask=%.2f%% area_reduction=%.2f%% alpha=%.2f suppression_strength=%.2f%% effective_suppression=%.2f%% components=%d",
            frame_count_,
            used_mask ? 1 : 0,
            mask_result.skipped ? 1 : 0,
            mask_result.raw_mask_ratio * 100.0,
            mask_result.final_mask_ratio * 100.0,
            mask_result.area_reduction_ratio * 100.0,
            mask_result.alpha,
            mask_result.suppression_strength * 100.0,
            mask_result.effective_suppression_ratio * 100.0,
            mask_result.components
          );
        } else {
          RCLCPP_INFO(
            this->get_logger(),
            "[Frame %d] dynamic mask disabled.",
            frame_count_
          );
        }
      }

      slam_->TrackStereo(left_img, right_img, t_left);

    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "stereoCallback error: %s", e.what());
    }
  }

private:
  std::unique_ptr<ORB_SLAM3::System> slam_;

  message_filters::Subscriber<ImageMsg> left_sub_;
  message_filters::Subscriber<ImageMsg> right_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Subscription<ImageMsg>::SharedPtr mask_sub_;

  std::mutex mask_mutex_;
  cv::Mat latest_mask_;
  double latest_mask_stamp_{0.0};
  bool has_mask_{false};

  bool use_dynamic_mask_{true};

  std::string left_topic_{"/camera/left/image_rect"};
  std::string right_topic_{"/camera/right/image_rect"};
  std::string mask_topic_{"/yolo/mask"};

  int mask_threshold_{1};
  int mask_dilate_size_{1};

  double mask_max_dt_{1.0};
  double max_stereo_dt_{0.08};

  double skip_mask_ratio_{0.75};

  double small_mask_w_{0.75};
  double small_mask_h_{0.90};

  double mid_mask_w_{0.65};
  double mid_mask_h_{0.82};

  double large_mask_w_{0.55};
  double large_mask_h_{0.72};

  double huge_mask_w_{0.45};
  double huge_mask_h_{0.62};

  double small_alpha_{0.65};
  double mid_alpha_{0.72};
  double large_alpha_{0.80};
  double huge_alpha_{0.88};

  int min_component_area_{80};

  int frame_count_{0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  if (argc < 3) {
    std::cerr << "Usage: dynamic_stereo_slam_node <vocab_path> <settings_path>" << std::endl;
    rclcpp::shutdown();
    return 1;
  }

  auto node = std::make_shared<StereoSlamNode>(argv[1], argv[2]);
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
 