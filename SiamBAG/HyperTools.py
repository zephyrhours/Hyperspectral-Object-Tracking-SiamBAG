# -*- coding: utf-8 -*-
import numpy as np
import os, io
import torch
from PIL import Image
from torchvision.transforms import CenterCrop
from torch.autograd import Variable
import torchvision.transforms as transforms
from channel_net import Channel_attention_net
from image_enhance import truncated_linear_stretch

r"""
Hyperspectral Toolbox from HOT_Tools
Author: Zephyr Hou
Time: 2022-8-25
Email: zephyrhou@126.com

All rights reserved!
"""


class ToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose(2, 0, 1)
        return torch.from_numpy(sample.astype(np.float32))


x_transforms = transforms.Compose([
    ToTensor()
])


# 读取XIMEA数据，转换原始png数据为高光谱数据
def X2Cube(img, B=[4, 4], skip=[4, 4], bands=16):
    """ A hyperspectral  datacube reconstruction function.

    Args:
        img (matrix):The input original XIMEA mosaic image, with the size of rows x cols.
        B (list or vector, optional): the size of mosaic window. Defaults to [4, 4].
        skip (list or vector, optional): the iterval of skip sampling, consistent with mosaic window. Defaults to [4, 4].
        bands (int, optional): the number of channels. Defaults to 16.

    Returns:
        DataCube (matrix): the datacube with the size of [rows/4]*[cols/4]*bands.

    Demo:
        hsi = X2Cube(np.array(img))
        hsi = X2Cube(np.array(img), B=[5, 5], skip=[5, 5], bands=25)
    """

    # Main Function
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indices
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M // skip[0], N // skip[1], bands)

    # # DataCube = DataCube.transpose(1, 0, 2)
    # DataCube = DataCube / DataCube.max() * 255  # 归一化
    # DataCube.astype('uint8')

    for i in range(bands):
        bandi = DataCube[:, :, i]
        DataCube[:, :, i] = (bandi - bandi.min()) / (bandi.max() - bandi.min()) * 255
    DataCube.astype('uint8')

    # DataCube = truncated_linear_stretch(DataCube, 2)

    return DataCube


def X2Cube2(img, B=[4, 4], skip=[4, 4], bands=16):
    # Main Function
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indices
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M // skip[0], N // skip[1], bands)

    return DataCube


def Cube2X(img, win=[4, 4]):
    img = np.array(img)
    rows, cols, bands = img.shape
    DataCube = np.zeros([4 * rows, 4 * cols])
    for i in range(rows):
        for j in range(cols):
            patch = img[i, j, :].reshape(win)
            DataCube[i * win[0]:(i + 1) * win[0], j * win[1]:(j + 1) * win[1]] = patch
    return DataCube


def folder_search(dir_path):
    """ A function to search the subfolder names under the specific direction.
    Args:
        dir_path (str): the search path.
    Returns:
        folder_list (list): file list containing all subfolders.
    Demo:
        folder_path = 'C:/Users/dream/Documents/Python/SiamFC_HOT/results/HOT2022/SiamFC'
        names = folder_search(folder_path)
        print(names)
    """
    # Main function
    folder_list = []
    subdir_list = os.listdir(dir_path)
    for i in subdir_list:
        file_path = os.path.join(dir_path, i)
        if os.path.isdir(file_path):  # judgment folders
            folder_list.append(i)
    return folder_list


def CenterCrop_HSI(hsi_path, crop_size):
    """Crops the given hyperspectral image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        hsi_path: the path of input image
        crop_size: sequence or int, desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    Returns:
         crop_hsi: Cropped hyperspectral image.

    """
    Img = Image.open(hsi_path)
    hsi = X2Cube(np.array(Img))
    hsi_tensor = torch.from_numpy(np.transpose(hsi, [2, 0, 1]))  # [bands, height, width]
    center_crop_hsi = CenterCrop(crop_size)  # rows, cols
    crop_hsi = center_crop_hsi(hsi_tensor)
    crop_hsi = np.transpose(np.array(crop_hsi), [1, 2, 0])  # [crop_height, crop_width, bands]

    return crop_hsi


def getPILCube(hsi_path):
    img1 = Image.open(hsi_path)  # .convert('RGB')
    img = X2Cube(np.array(img1))  # 16通道
    img = img.transpose(1, 0, 2)
    img = img / img.max() * 255  # 归一化
    img.astype('uint8')

    tt = img[:, :, -1, np.newaxis]
    tArr = [Image.fromarray(img[:, :, kk * 3:kk * 3 + 3].astype('uint8')).convert('RGB') for kk in range(5)]
    img = np.concatenate((tArr[0], tArr[1], tArr[2], tArr[3], tArr[4], tt), axis=2)

    return img


def _split_Channel(feat_channel, order):
    bands = order.numel()
    splitNum = bands // 3
    orderR = order[0, 0:bands // 3]
    orderG = order[0, bands // 3:(bands // 3) * 2]
    orderB = order[0, (bands // 3) * 2:(bands - 1)]
    res = []
    b = feat_channel.size()[0]

    for i in range(splitNum):
        gg = feat_channel[None, 0, [orderR[i], orderG[splitNum - 1 - i], orderB[i]], :, :]
        for k in range(1, b):
            gg = torch.cat((gg, feat_channel[None, k, [orderR[i], orderG[splitNum - 1 - i], orderB[i]], :, :]), dim=0)
        res.append(gg)

    # 最后一个波段与第一个波段和中间的波段构成第六组数据
    gg = feat_channel[None, 0, order[0, [0, bands // 2, bands - 1]], :, :]
    res.append(gg)
    return res


def getHsiFrame(frame, model_path=''):  # [height,width,bands]
    hsiImage = X2Cube(np.array(frame))
    exemplar_img = x_transforms(hsiImage)[None, :, :, :]
    hsiImage_var = Variable(exemplar_img)
    # print('hsiImage_var.size:', hsiImage_var.size())
    model = Channel_attention_net()
    model.load_state_dict(torch.load(model_path))
    # res, orderY = model(hsiImage_var)  # 1,16,xx,xx
    _, orderY = model(hsiImage_var)  # 1,16,xx,xx
    res = hsiImage_var
    order = orderY[1]  # ordered bands index
    order_weight = orderY[0]  # ordered bands weight
    nums_group = order.numel() // 3 + 1
    res = _split_Channel(res, order)

    for i in range(nums_group):
        if i == 0:
            hsiFrame = res[0]
        else:
            hsiFrame = torch.cat((hsiFrame, res[i]), 1)

    hsiFrame = torch.squeeze(hsiFrame)
    hsiFrame = hsiFrame.detach().numpy()
    hsiFrame = hsiFrame.transpose(1, 2, 0)
    hsiFrameArr = []
    for i in range(order.numel() // 3 + 1):
        hsiFrameArr.append(hsiFrame[:, :, i * 3:i * 3 + 3])
    return hsiFrameArr, order_weight  # the size is [height,width,bands]
