from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
# from imutils.video import FPS, WebcamVideoStream
import argparse
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd


#输入模型的绝对地址、修改50line的图片地址、60line的类别数
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default="/media/vs/qi/data/ssd.pytorch/weights/ssd_VOC_500.pth",
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                            FONT, 1, (255, 0, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    frame = cv2.imread('/media/vs/qi/data/ssd.pytorch/doc/helmet01.jpg')
    frame = predict(frame)
    window_name = 'Object detector'
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 640, 640)
    cv2.imshow(window_name, frame)
    # cv2.imshow('frame', frame)


if __name__ == '__main__':
    net = build_ssd('test', 300, 3)  # initialize SSD ， 3为类别数
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

    cv2_demo(net.eval(), transform)
    # Press any key to close the image
    cv2.waitKey()
    # cleanup
    cv2.destroyAllWindows()