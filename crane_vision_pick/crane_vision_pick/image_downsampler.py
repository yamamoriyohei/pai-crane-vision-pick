"""カメラ画像を間引いて小さくするノード.

/image_raw (640x480, rgb8, 10Hz) → /camera/image_small (既定 320x240)

第3回講義の「保存すると取得Hzが落ちる / 画像は大きい」への対策:
  640x480x3 = 921,600 B/枚 → 10Hz で約 9.2 MB/s。100 エピソード x 12 s で 11 GB になる。
  1/2 に間引くと 1/4 の約 2.3 MB/s。学習もこの解像度で行うので、記録の時点で縮小しても学習結果は変わらない。
cv_bridge を使わず numpy のスライスで間引くので依存が少なく速い。
"""
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

CHANNELS = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}


class ImageDownsampler(Node):
    def __init__(self):
        super().__init__('image_downsampler')
        self.declare_parameter('factor', 2, ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('input', '/image_raw')
        self.declare_parameter('output', '/camera/image_small')
        self.factor = int(self.get_parameter('factor').value)
        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(Image, self.get_parameter('output').value, qos)
        self.create_subscription(Image, self.get_parameter('input').value, self.cb, qos)
        self.warned = False

    def cb(self, msg: Image):
        ch = CHANNELS.get(msg.encoding)
        if ch is None:
            if not self.warned:
                self.get_logger().error(f'unsupported encoding: {msg.encoding}')
                self.warned = True
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
        f = self.factor
        small = np.ascontiguousarray(img[::f, ::f, :3] if ch >= 3 else img[::f, ::f])
        out = Image()
        out.header = msg.header          # 撮影時刻（シミュレーション時間）をそのまま引き継ぐ
        out.height, out.width = small.shape[:2]
        out.encoding = 'rgb8' if msg.encoding in ('rgb8', 'rgba8') else (
            'bgr8' if msg.encoding in ('bgr8', 'bgra8') else 'mono8')
        out.is_bigendian = 0
        out.step = out.width * (small.shape[2] if small.ndim == 3 else 1)
        out.data = small.tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ImageDownsampler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
