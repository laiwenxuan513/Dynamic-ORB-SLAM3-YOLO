# Dynamic-ORB-SLAM3-YOLO
基于 RK3588 嵌入式平台的动态场景视觉 SLAM，搭载 NPU 加速 YOLO 动态特征剔除算法

## 项目简介
本项目针对室内动态场景下传统ORB-SLAM3位姿漂移、建图精度差问题，采用YOLO实例分割网络实时识别行人、移动物体，在特征提取阶段剔除动态区域特征点，实现高鲁棒性静态地图构建与精准定位；整套算法部署于瑞芯微RK3588嵌入式开发板，搭配双目相机完成轻量化实时运行，适用于室内机器人导航场景。

## 硬件平台
- 主控：RK3588开发板
- 传感器：双目同步相机
- 推理加速：RKNN NPU硬件加速YOLO分割模型

## 软件环境依赖
1. ROS2 Humble
2. OpenCV 4.5+
3. RKNN Toolkit 2
4. Python3.8 / PyTorch
5. Eigen、DBoW3、OpenGV（ORB-SLAM3依赖库）
