import cv2
import numpy as np


# RGB图像全局直方图均衡化
def hisEqulColor_global(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])   # 对第1个通道即亮度通道进行全局直方图均衡化并保存
    ycrcb = cv2.merge(channels)
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img


# RGB图像进行自适应直方图均衡化，代码同上的地方不再添加注释
def hisEqulColor_adaptive(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])

    ycrcb = cv2.merge(channels)
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img


def truncated_linear_stretch(image, truncated_value, max_out=255, min_out=0):
    def gray_process(img):
        truncated_down = np.percentile(img, truncated_value)
        truncated_up = np.percentile(img, 100 - truncated_value)
        img = (img - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
        img[img < min_out] = min_out
        img[img > max_out] = max_out
        if max_out <= 255:
            img = np.uint8(img)
        elif max_out <= 65535:
            img = np.uint16(img)
        return img

    #  如果是多波段
    if len(image.shape) == 3:
        image_stretch = []
        for i in range(image.shape[0]):
            gray = gray_process(image[i])
            image_stretch.append(gray)
        image_stretch = np.array(image_stretch)
    #  如果是单波段
    else:
        image_stretch = gray_process(image)

    return image_stretch

# def shadow_remove(img):
