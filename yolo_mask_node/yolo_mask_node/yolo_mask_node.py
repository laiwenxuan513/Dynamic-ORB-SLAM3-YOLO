import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import time
import threading

from rknnlite.api import RKNNLite


# =========================
# Basic config
# =========================
# 指定模型路径
# 指定订阅哪个图像 topic
# 指定发布哪些 topic
# 设置 YOLO 输入尺寸
# 设置置信度阈值
# 设置 NMS 阈值
# 限制候选框数量
MODEL_PATH = '/home/elf/rknn_model_zoo/examples/yolov5/model/yolov5s_relu.rknn'

IMAGE_TOPIC = '/camera/left/image_raw'
DEBUG_TOPIC = '/yolo/debug_image'
MASK_TOPIC = '/yolo/mask'

IMG_SIZE = 640

# 这里不要太低。
# 之前 0.25 导致几千个假 person 框，后处理直接爆炸。
OBJ_THRESH = 0.50
NMS_THRESH = 0.45

# NMS 前最多保留多少候选框。
# 防止误检过多导致 NMS 极慢。
MAX_NMS_BOXES = 300

# 最终最多画多少个框。
# 正常画面里 person 不可能有几千个。
MAX_DRAW_BOXES = 20

# COCO class 0 = person
PERSON_CLASS_ID = 0

# YOLOv5 anchors
ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],
    [[30, 61], [62, 45], [59, 119]],
    [[116, 90], [156, 198], [373, 326]],
]

STRIDES = [8, 16, 32]


# =========================
# Utility functions
# 把原始图像按比例缩放到 640x640
# 不足区域用灰色 padding 补齐
# =========================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize image with unchanged aspect ratio using padding.
    Return:
        img: letterboxed image
        r: resize ratio
        pad: (pad_x, pad_y)
    """
    shape = im.shape[:2]  # h, w

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w, h

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    im = cv2.copyMakeBorder(
        im,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return im, r, (left, top)


#RKNN 输出格式整理
def normalize_output(out):
    """
    Convert RKNN output to shape: H, W, 3, 85
    Compatible with common YOLOv5 RKNN output formats.
    """
    out = np.squeeze(out)

    # case 1: 255 x H x W
    if out.ndim == 3 and out.shape[0] == 255:
        c, h, w = out.shape
        out = out.reshape(3, 85, h, w)
        out = out.transpose(2, 3, 0, 1)  # H W 3 85
        return out

    # case 2: H x W x 255
    if out.ndim == 3 and out.shape[-1] == 255:
        h, w, c = out.shape
        out = out.reshape(h, w, 3, 85)
        return out

    # case 3: 3 x H x W x 85
    if out.ndim == 4 and out.shape[0] == 3 and out.shape[-1] == 85:
        out = out.transpose(1, 2, 0, 3)
        return out

    # case 4: H x W x 3 x 85
    if out.ndim == 4 and out.shape[-1] == 85:
        return out

    raise RuntimeError(f'Unsupported YOLO output shape: {out.shape}')

#解码person
def decode_person_outputs(outputs):
    """
    Decode YOLOv5 RKNN outputs.
    Only decode person class to reduce computation.

    Important:
    - Only keep class 0: person
    - Apply higher threshold
    - Limit candidates before NMS
    """
    all_boxes = []
    all_scores = []

    for i, out in enumerate(outputs):
        pred = normalize_output(out)

        h, w, _, _ = pred.shape
        stride = STRIDES[i]
        anchors = np.array(ANCHORS[i], dtype=np.float32)

        pred = sigmoid(pred)

        obj_conf = pred[..., 4]

        # COCO class 0 = person.
        # YOLO output layout: [x, y, w, h, obj, cls0, cls1, ...]
        person_conf = pred[..., 5]

        scores = obj_conf * person_conf

        valid = scores > OBJ_THRESH

        if not np.any(valid):
            continue

        # Only calculate grid/xy/wh after filtering score.
        # This avoids doing too much work when most boxes are invalid.
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        grid = np.stack((grid_x, grid_y), axis=-1)
        grid = np.expand_dims(grid, axis=2)  # H W 1 2

        xy = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
        wh = (pred[..., 2:4] * 2.0) ** 2 * anchors.reshape(1, 1, 3, 2)

        xy = xy[valid]
        wh = wh[valid]
        scores_keep = scores[valid]

        boxes = np.zeros((xy.shape[0], 4), dtype=np.float32)
        boxes[:, 0] = xy[:, 0] - wh[:, 0] / 2
        boxes[:, 1] = xy[:, 1] - wh[:, 1] / 2
        boxes[:, 2] = xy[:, 0] + wh[:, 0] / 2
        boxes[:, 3] = xy[:, 1] + wh[:, 1] / 2

        all_boxes.append(boxes)
        all_scores.append(scores_keep.astype(np.float32))

    if len(all_boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    # Limit number of boxes before NMS.
    # NMS complexity is high when there are too many boxes.
    if len(scores) > MAX_NMS_BOXES:
        idx = np.argsort(scores)[-MAX_NMS_BOXES:]
        boxes = boxes[idx]
        scores = scores[idx]

    return boxes, scores

#去掉重叠的重复框，只保留置信度最高的框
def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        ww = np.maximum(0.0, xx2 - xx1)
        hh = np.maximum(0.0, yy2 - yy1)

        inter = ww * hh
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


# =========================
# ROS2 Node
# 1. 创建 CvBridge
# 2. 加载 RKNN 模型
# 3. 初始化 RKNN runtime
# 4. 订阅 /camera/left/image_raw
# 5. 创建 /yolo/debug_image 发布器
# 6. 创建 /yolo/mask 发布器
# 7. 启动 YOLO worker 线程

# =========================

class YoloMaskNode(Node):
    def __init__(self):
        super().__init__('yolo_mask_node')

        self.bridge = CvBridge()

        self.latest_frame = None
        self.latest_header = None
        self.latest_id = 0
        self.processed_id = 0

        self.lock = threading.Lock()
        self.running = True
        self.infer_count = 0

        self.get_logger().info('Loading RKNN model...')
        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(MODEL_PATH)
        if ret != 0:
            self.get_logger().error(f'Failed to load RKNN model: {MODEL_PATH}')
            raise RuntimeError('load_rknn failed')

        self.get_logger().info('Initializing RKNN runtime...')

        # Try RK3588 3-core NPU first.
        # If it fails, fallback to default runtime.
        try:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret == 0:
                self.get_logger().info('RKNN runtime initialized with NPU_CORE_0_1_2')
            else:
                self.get_logger().warn(
                    f'init_runtime with NPU_CORE_0_1_2 failed, ret={ret}, fallback to default'
                )
                ret = self.rknn.init_runtime()
        except Exception as e:
            self.get_logger().warn(f'init_runtime core_mask failed: {e}, fallback to default')
            ret = self.rknn.init_runtime()

        if ret != 0:
            self.get_logger().error('Failed to init RKNN runtime')
            raise RuntimeError('init_runtime failed')

        self.get_logger().info(f'RKNN model loaded: {MODEL_PATH}')

        self.sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10
        )

        self.debug_pub = self.create_publisher(Image, DEBUG_TOPIC, 10)
        self.mask_pub = self.create_publisher(Image, MASK_TOPIC, 10)

        self.worker = threading.Thread(target=self.inference_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(f'Subscribe: {IMAGE_TOPIC}')
        self.get_logger().info(f'Publish debug: {DEBUG_TOPIC}')
        self.get_logger().info(f'Publish mask: {MASK_TOPIC}')
        self.get_logger().info('YOLO worker thread started')



# =========================
# 1. 接收 ROS Image
# 2. 转成 OpenCV 图像
# 3. 如果是 mono8，就转成 BGR
# 4. 保存为 latest_frame
# =========================
    def image_callback(self, msg):
        """
        Callback only stores latest frame.
        Heavy YOLO inference is done in worker thread.
        """
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # Your camera image may be mono8.
        # Convert grayscale to BGR for YOLO input and debug drawing.
        if len(img.shape) == 2:
            frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            frame = img.copy()

        with self.lock:
            self.latest_frame = frame
            self.latest_header = msg.header
            self.latest_id += 1

 # =========================
# 1. 检查有没有最新帧
# 2. 如果有新帧，复制出来
# 3. 调用 run_yolo()
# 4. 发布 debug 图
# 5. 发布 mask 图
# 6. 打印耗时
# 相机线程负责收图
# YOLO线程负责推理
# =========================
    def inference_loop(self):
        while self.running and rclpy.ok():
            with self.lock:
                if self.latest_frame is None or self.latest_id == self.processed_id:
                    frame = None
                    header = None
                    frame_id = self.latest_id
                else:
                    frame = self.latest_frame.copy()
                    header = self.latest_header
                    frame_id = self.latest_id

            if frame is None:
                time.sleep(0.005)
                continue

            self.processed_id = frame_id

            try:
                debug, mask, persons, timing = self.run_yolo(frame)
            except Exception as e:
                self.get_logger().error(f'YOLO processing error: {e}')
                time.sleep(0.01)
                continue

            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
                debug_msg.header = header
                self.debug_pub.publish(debug_msg)

                mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                mask_msg.header = header
                self.mask_pub.publish(mask_msg)
            except Exception as e:
                self.get_logger().error(f'publish error: {e}')
                continue

            self.infer_count += 1

            if self.infer_count % 5 == 0:
                self.get_logger().info(
                    f'infer_count={self.infer_count}, persons={persons}, '
                    f'pre={timing["pre"]:.1f}ms, '
                    f'infer={timing["infer"]:.1f}ms, '
                    f'post={timing["post"]:.1f}ms, '
                    f'draw={timing["draw"]:.1f}ms, '
                    f'total={timing["total"]:.1f}ms'
                )
# =========================
# 原图 frame
# ↓
# letterbox 到 640x640
# ↓
# BGR 转 RGB
# ↓
# expand dims 变成 NHWC
# ↓
# RKNN inference
# ↓
# decode person
# ↓
# NMS
# ↓
# 坐标映射回原图
# ↓
# 生成 debug 图
# ↓
# 生成 mask
# ↓
# 返回结果和耗时
 # =========================

    def run_yolo(self, frame):
        h0, w0 = frame.shape[:2]

        t0 = time.time()

        # Preprocess
        img_lb, ratio, pad = letterbox(frame, new_shape=(IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0)

        t1 = time.time()

        # RKNN inference
        outputs = self.rknn.inference(inputs=[input_data])

        t2 = time.time()

        # Decode only person class
        boxes, scores = decode_person_outputs(outputs)

        # NMS
        if len(boxes) > 0:
            keep = nms(boxes, scores, NMS_THRESH)
            if len(keep) > 0:
                boxes = boxes[keep]
                scores = scores[keep]
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)

        # Limit final boxes
        if len(scores) > MAX_DRAW_BOXES:
            idx = np.argsort(scores)[-MAX_DRAW_BOXES:]
            boxes = boxes[idx]
            scores = scores[idx]

        t3 = time.time()

        # Draw debug image and generate mask
        pad_x, pad_y = pad

        debug = frame.copy()
        mask = np.zeros((h0, w0), dtype=np.uint8)

        persons = 0

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box

            x1 = int((x1 - pad_x) / ratio)
            y1 = int((y1 - pad_y) / ratio)
            x2 = int((x2 - pad_x) / ratio)
            y2 = int((y2 - pad_y) / ratio)

            x1 = max(0, min(w0 - 1, x1))
            y1 = max(0, min(h0 - 1, y1))
            x2 = max(0, min(w0 - 1, x2))
            y2 = max(0, min(h0 - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            persons += 1

            # Expand mask slightly for SLAM dynamic filtering
            expand = 8
            mx1 = max(0, x1 - expand)
            my1 = max(0, y1 - expand)
            mx2 = min(w0 - 1, x2 + expand)
            my2 = min(h0 - 1, y2 + expand)

            cv2.rectangle(mask, (mx1, my1), (mx2, my2), 255, -1)

            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                debug,
                f'person {score:.2f}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        t4 = time.time()

        timing = {
            'pre': (t1 - t0) * 1000.0,
            'infer': (t2 - t1) * 1000.0,
            'post': (t3 - t2) * 1000.0,
            'draw': (t4 - t3) * 1000.0,
            'total': (t4 - t0) * 1000.0,
        }

        return debug, mask, persons, timing

    def destroy_node(self):
        self.running = False

        try:
            if hasattr(self, 'worker') and self.worker.is_alive():
                self.worker.join(timeout=1.0)
        except Exception:
            pass

        try:
            self.rknn.release()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = YoloMaskNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

