from __future__ import absolute_import, division

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import namedtuple
from got10k.trackers import Tracker
from image_enhance import truncated_linear_stretch, hisEqulColor_global, hisEqulColor_adaptive


class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 192, 11, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512))

        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)  # kernal_size = 3, stride =1
        self.conv_reg_x = nn.Conv2d(512, 512, 3)  # kernal_size = 3
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)  # kernal_size = 3, stride =1
        self.conv_cls_x = nn.Conv2d(512, 512, 3)  # kernal_size = 3
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)  # kernal_size = 1

    def forward(self, z, x):  # z:examplar image, x:instance image
        return self.inference(x, **self.learn(z))

    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 512, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 512, k, k)

        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)

        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        out_cls = F.conv2d(x_cls, kernel_cls)

        return out_reg, out_cls


class TrackerSiamBAG(Tracker):
    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamBAG, self).__init__(
            name='SiamBAG', is_deterministic=True)

        # parameters initialization
        self.use_scale_tune = True
        self.anchors = None
        self.avg_color = None
        self.center = None
        self.cfg = None
        self.hann_window2 = None
        self.hann_window = None
        self.kernelArr = None
        self.kernel_cls_Arr = None
        self.kernel_reg_Arr = None
        self.response_sz = None
        self.scales_y = None
        self.scales_x = None
        self.scale_factors = None
        self.target_sz = None
        self.upscale_sz = None
        self.x_sz = None
        self.z_sz = None
        self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamRPN()

        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))  # GPU -> CPU
        self.net = self.net.to(self.device)  # network -> GPU

    def parse_args(self, **kargs):
        self.cfg = {
            'exemplar_sz': 127,
            'instance_sz': 271,
            'context': 0.5,
            'total_stride': 8,
            'ratios': [0.33, 0.5, 1, 2, 3],
            'scales': [8, ],
            'penalty_k': 0.055,        # 0.055
            'window_influence': 0.42,  # 0.42
            'lr': 0.295,               # 0.295
            # for scale fine-turning parameters
            'response_sz': 17,
            'response_up': 16,
            'scale_lr': 0.59,  # 0.59
            'scale_penalty': 0.9745,  # 0.9745
            'window_influence2': 0.176,  # 0.176
            'scale_factors': [0.98, 1, 1.02],   # [0.98, 1, 1.02] best
        }

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)

        print('Scale Tune: ', self.use_scale_tune)

    def init(self, imageArr, box):  # the input box is [x,y,w,h]
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)  # [y, x, h, w]
        self.center, self.target_sz = box[:2], box[2:]

        # for small target, use larger search region
        if np.prod(self.target_sz) / np.prod(np.array(imageArr[0]).shape[:2]) < 0.007:  # 0.0004
            self.cfg = self.cfg._replace(instance_sz=287)

        # generate anchors
        self.response_sz = (self.cfg.instance_sz - self.cfg.exemplar_sz) // self.cfg.total_stride + 1  # 19
        self.anchors = self._create_anchors(self.response_sz)  #

        # create hanning window
        self.hann_window = np.outer(
            np.hanning(self.response_sz),
            np.hanning(self.response_sz))
        self.hann_window = np.tile(
            self.hann_window.flatten(),
            len(self.cfg.ratios) * len(self.cfg.scales))

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.kernel_reg_Arr = []
        self.kernel_cls_Arr = []
        self.kernelArr = []
        for image in imageArr:
            image = np.asarray(image)
            # global image enhancement strategy
            image = hisEqulColor_global(image)  # 0.641

            self.avg_color = np.mean(image, axis=(0, 1))
            exemplar_image = self._crop_and_resize(
                image, self.center, self.z_sz,
                self.cfg.exemplar_sz, self.avg_color)

            # local image enhance
            # exemplar_image = hisEqulColor_adaptive(exemplar_image)  # 0.636

            # classification and regression kernels
            exemplar_image = torch.from_numpy(exemplar_image).to(
                self.device).permute([2, 0, 1]).unsqueeze(0).float()
            with torch.set_grad_enabled(False):  # 关闭求导
                self.net.eval()
                kernel_reg, kernel_cls = self.net.learn(exemplar_image)
                self.kernel_reg_Arr.append(kernel_reg)
                self.kernel_cls_Arr.append(kernel_cls)

                kernel = self.net.feature(exemplar_image)
                self.kernelArr.append(kernel)

    def update(self, imageArr, order_weight, img_files, f, boxes):
        # All bands are grouped into several RGB pseudo-color images according to the weight
        bands = order_weight.numel()
        order_weight_Arr = torch.empty(1, bands//3+1)
        for i in range(bands//3+1):
            if i < bands//3:
                order_weight_Arr[0, i] = order_weight[0, i] + order_weight[0, (bands // 3) * 2 - 1 - i] + \
                                         order_weight[0, (bands // 3) * 2 + i]
            else:
                order_weight_Arr[0, i] = order_weight[0, [0]] + order_weight[0, [bands // 2]] + \
                                         order_weight[0, [bands-1]]

        order_weight_Arr = torch.squeeze(order_weight_Arr)
        order_weight_Arr = order_weight_Arr.detach().numpy()
        order_weight_Arr = order_weight_Arr / np.sum(order_weight_Arr)

        splitNum = 0
        for image in imageArr:
            image = np.asarray(image)
            # global image enhancement strategy
            image = hisEqulColor_global(image)  # 0.641

            instance_image = self._crop_and_resize(
                image, self.center, self.x_sz,
                self.cfg.instance_sz, self.avg_color)

            # local image enhancement strategy
            # instance_image = hisEqulColor_adaptive(instance_image)  # 0.636

            # classification and regression outputs
            instance_image = torch.from_numpy(instance_image).to(
                self.device).permute(2, 0, 1).unsqueeze(0).float()

            with torch.set_grad_enabled(False):
                self.net.eval()
                out_reg, out_cls = self.net.inference(
                    instance_image, self.kernel_reg_Arr[splitNum], self.kernel_cls_Arr[splitNum])

            # offsets (dx,dy,dw,dh)  anchors(cx,cy,w,h)
            offsets = out_reg.permute(
                1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()  # 4x1805(1805=19*19*5) [dx,dy,dw,dh]
            offsets[0] = offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]  # dx
            offsets[1] = offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]  # dy
            offsets[2] = np.exp(offsets[2]) * self.anchors[:, 2]  # dw
            offsets[3] = np.exp(offsets[3]) * self.anchors[:, 3]  # dh

            # scale and ratio penalty
            penalty = self._create_penalty(self.target_sz, offsets)

            # response(background, positive or negative)
            response = F.softmax(out_cls.permute(
                1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()  # (1805，)  # odd channel is positive
            response = response * penalty
            response = (1 - self.cfg.window_influence) * response + \
                       self.cfg.window_influence * self.hann_window

            if splitNum == 0:
                response_Arr = response * order_weight_Arr[splitNum]   # weighted fusion
                offsets_Arr = offsets
            else:
                response_Arr = response_Arr + response * order_weight_Arr[splitNum]
                # response_Arr = response_Arr + response
                offsets_Arr = offsets_Arr + offsets

            splitNum += 1

        # peak location
        best_id = np.argmax(response_Arr)
        offset = offsets_Arr[:, best_id] * self.z_sz / self.cfg.exemplar_sz
        offset = offset / splitNum

        # update center
        self.center += offset[:2][::-1]  # 坐标转换 (x,y) -> (y, x)
        self.center = np.clip(self.center, 0, image.shape[:2])

        # update scale
        lr = response_Arr[best_id] * self.cfg.lr
        self.target_sz = (1 - lr) * self.target_sz + lr * offset[2:][::-1]  # 坐标转换，(w，h)-> (h,w)
        self.target_sz = np.clip(self.target_sz, 10, image.shape[:2])

        # update exemplar and instance sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # ###whether performance scale fine-tuning strategy###
        if self.use_scale_tune is False:  # Perform SiamRPN operation
            # return 1-indexed and left-top based bounding box
            box = np.array([
                np.round(self.center[1] + 1 - (self.target_sz[1] - 1) / 2),
                np.round(self.center[0] + 1 - (self.target_sz[0] - 1) / 2),
                np.round(self.target_sz[1]), np.round(self.target_sz[0])])
            return box  # （x,y,w,h）

        else:  # Perform SiamBAG operation for scale fine-turning
            self.scale_factors = np.array(self.cfg.scale_factors)  # 0.648
            scale_num = len(self.cfg.scale_factors)

            self.upscale_sz = self.cfg.response_up * self.cfg.response_sz  # 16*17=272
            self.hann_window2 = np.outer(
                np.hanning(self.upscale_sz),
                np.hanning(self.upscale_sz))
            self.hann_window2 /= self.hann_window2.sum()

            # max_idx = np.argmax(order_weight_Arr)  # obtain a three-channel image with maximum information
            max_idx = 0
            image = np.asarray(imageArr[max_idx])
            self.avg_color = np.mean(image, axis=(0, 1))

            instance_images = [self._crop_and_resize(
                image, self.center, self.x_sz * f,
                out_size=self.cfg.instance_sz,
                pad_color=self.avg_color) for f in self.scale_factors]

            instance_images = np.stack(instance_images, axis=0)
            instance_images = torch.from_numpy(np.array(instance_images)).to(
                self.device).permute([0, 3, 1, 2]).float()

            # responses
            with torch.set_grad_enabled(False):
                self.net.eval()
                instances2 = self.net.feature(instance_images)
                responses2 = F.conv2d(instances2, self.kernelArr[max_idx]) * 0.001  # 卷积操作，论文中的互相关
            responses2 = responses2.squeeze(1).cpu().numpy()

            # up-sample responses and penalize scale changes
            responses2 = np.stack([cv2.resize(
                t, (self.upscale_sz, self.upscale_sz),  # (272,272)
                interpolation=cv2.INTER_CUBIC) for t in responses2], axis=0)
            responses2[:scale_num // 2] *= self.cfg.scale_penalty
            responses2[scale_num // 2 + 1:] *= self.cfg.scale_penalty

            # peak scale
            responses2 *= self.hann_window2
            scale_id = np.argmax(np.amax(responses2, axis=(1, 2)))
            # peak location
            response2 = responses2[scale_id]
            response2 -= response2.min()
            response2 /= response2.sum() + 1e-16
            response2 = (1 - self.cfg.window_influence2) * response2 + self.cfg.window_influence2 * self.hann_window2
            loc = np.unravel_index(response2.argmax(), response2.shape)

            # locate target center
            disp_in_response = np.array(loc) - self.upscale_sz // 2
            disp_in_instance = disp_in_response * self.cfg.total_stride / self.cfg.response_up
            disp_in_image = disp_in_instance * self.x_sz * self.scale_factors[scale_id] / self.cfg.instance_sz
            self.center += disp_in_image

            # target size
            scale = (1 - self.cfg.scale_lr) * 1.0 + self.cfg.scale_lr * self.scale_factors[scale_id]
            self.target_sz *= scale
            self.z_sz *= scale
            self.x_sz *= scale

            # return 1-indexed and left-top based bounding box
            box = np.array([
                np.round(self.center[1] + 1 - (self.target_sz[1] - 1) / 2),
                np.round(self.center[0] + 1 - (self.target_sz[0] - 1) / 2),
                np.round(self.target_sz[1]), np.round(self.target_sz[0])])

            return box  # （x,y,w,h）

    def _create_anchors(self, response_sz):  # response_sz = 19
        anchor_num = len(self.cfg.ratios) * len(self.cfg.scales)  # 5*1=5
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)

        size = self.cfg.total_stride * self.cfg.total_stride  # 8*8=64, anchor's area
        ind = 0
        for ratio in self.cfg.ratios:  # [0.33, 0.5, 1, 2, 3]
            w = int(np.sqrt(size / ratio))
            h = int(w * ratio)  # w*h=s^2,保证生成的anchor的面积不变，这里anchor面积就是size=64
            for scale in self.cfg.scales:  # [8,]
                anchors[ind, 0] = 0
                anchors[ind, 1] = 0
                anchors[ind, 2] = w * scale  # 设置不同scales下的anchor 宽高大小
                anchors[ind, 3] = h * scale  # 因为这里只有一个尺度，所以每个anchor boxes面积大小为64*8=512
                ind += 1  # anchor boxes面积相同，但是长宽不同，这里共有5种尺度
        anchors = np.tile(anchors, response_sz * response_sz).reshape((-1, 4))  # 沿着x轴将anchors,平铺19*19遍,response_sz=19
        # 这里实际上就是将图片分成了19*19的网格，每个grid生成5种尺度的anchor boxes，而每个anchor boxes又有四个坐标（cx,cy,w,h）
        # 生成19*19=361个grids，生成361*5=1805个大小不同的anchor boxes.

        begin = -(response_sz // 2) * self.cfg.total_stride  # -9*8=-72
        xs, ys = np.meshgrid(
            begin + self.cfg.total_stride * np.arange(response_sz),
            begin + self.cfg.total_stride * np.arange(response_sz))  # xs,ys:19x19
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)  # anchor的中心点坐标 cx
        anchors[:, 1] = ys.astype(np.float32)  # anchor的中心点坐标 cy

        return anchors

    def _create_penalty(self, target_sz, offsets):
        def padded_size(w, h):
            context = self.cfg.context * (w + h)
            return np.sqrt((w + context) * (h + context))

        def larger_ratio(r):
            return np.maximum(r, 1 / r)

        src_sz = padded_size(
            *(target_sz * self.cfg.exemplar_sz / self.z_sz))
        dst_sz = padded_size(offsets[2], offsets[3])
        change_sz = larger_ratio(dst_sz / src_sz)

        src_ratio = target_sz[1] / target_sz[0]
        dst_ratio = offsets[2] / offsets[3]
        change_ratio = larger_ratio(dst_ratio / src_ratio)

        penalty = np.exp(-(change_ratio * change_sz - 1) * self.cfg.penalty_k)

        return penalty

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch