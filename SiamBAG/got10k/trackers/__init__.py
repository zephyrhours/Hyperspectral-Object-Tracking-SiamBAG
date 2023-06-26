from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from HyperTools import X2Cube, getHsiFrame
from ..utils.viz import show_frame
import copy


class Tracker(object):

    def __init__(self, name, is_deterministic=False):
        self.name = name
        self.is_deterministic = is_deterministic

    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image, order_weight, img_files, f, boxes):
        raise NotImplementedError()

    def track(self, img_files, box, sub_dir='RGB', visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        if sub_dir == 'HSI':
            channel_model_path = './pretrained/final_whisper_35_videos_channel.pth'
            for f, img_file in enumerate(img_files):  # f为索引，img_file为包含具体路径的文件名
                imageMosaic = Image.open(img_file)
                imageArr, order_weight = getHsiFrame(imageMosaic, channel_model_path)  # [height,width,bands]
                imageArr = [Image.fromarray(image.astype('uint8')).convert('RGB') for image in imageArr]

                start_time = time.time()
                if f == 0:
                    self.init(imageArr, box)
                else:
                    boxes[f, :] = self.update(imageArr, order_weight, img_files, f, boxes)
                times[f] = time.time() - start_time

                if visualize:
                    false_color_file = copy.deepcopy(img_file)
                    false_color_file = false_color_file.replace('HSI', 'HSI-FalseColor')
                    false_color_file = false_color_file.replace('.png', '.jpg')
                    image = Image.open(false_color_file)

                    seq_names = false_color_file.split("\\")[-3]
                    # fig_name = self.name + ':' + seq_names
                    fig_name = self.name
                    show_frame(image, boxes[f, :], num_frames=frame_num, frame_idx=f, fig_name=fig_name)

                    # imageHSI = X2Cube(np.array(imageMosaic))
                    # imageRGB = imageHSI[:, :, [15, 9, 8]].astype(np.uint8)
                    # show_frame(imageRGB, boxes[f, :], num_frames=frame_num, frame_idx=f)
        else:
            for f, img_file in enumerate(img_files):
                image = Image.open(img_file)
                if not image.mode == 'RGB':
                    image = image.convert('RGB')

                start_time = time.time()
                if f == 0:
                    self.init(image, box)
                else:
                    boxes[f, :] = self.update(image)
                times[f] = time.time() - start_time

                if visualize:
                    show_frame(image, boxes[f, :], num_frames=frame_num, frame_idx=f)

        return boxes, times


from .identity_tracker import IdentityTracker
