from PIL import Image
from pathlib import Path
import cv2
import os

def resize_image(image, target_size=(45, 70)):
    # 使用插值方法将图像缩放到目标大小
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image
    # r = resize_image(img)
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Resized Image', r)
    # cv2.waitKey(0)



import cv2
import numpy as np

def select_color_range(image, target_color=(223, 223, 221), tolerance=5):
    # 将目标颜色转换为numpy数组
    target_color = np.array(target_color, dtype=np.uint8)

    # 设置上下浮动的范围
    lower_bound = target_color - tolerance
    upper_bound = target_color + tolerance

    # 使用cv2.inRange选择颜色范围内的像素
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # 将原图像和掩码进行与运算，得到选中的像素
    selected_pixels = cv2.bitwise_and(image, image, mask=mask)

    return mask, selected_pixels
    # # 显示原图像、掩码和选中的像素
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Selected Pixels', selected_pixels)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def segment_img(img, colored_img):
    seg = []

    # cv2.imshow("Original", img)
    ret, thresh = cv2.threshold(img, 175, 200, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)

    kernel = np.ones((4,4),dtype="uint8") # 查以某点为中心 6*6 区域内是不是全是这个颜色。这个超参根据需要调整

    eroded = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel=kernel)
    # cv2.imshow("eroded", eroded)

    non_black_rows = np.any(eroded > 128, axis=1)
    # 取最大连续的非全黑行
    l = 0
    r = -1
    ll = -1
    for p, i in enumerate(non_black_rows):
        if i:
            if ll == -1:
                ll = p
        else:
            if ll != -1:
                if p - ll > r - l:
                    r = p
                    l = ll # 左闭右开区间
                ll = -1
    p = len(non_black_rows)
    if ll != -1: # 末尾处理
        if p - ll > r - l:
            r = p
            l = ll
        ll = -1

    row_clamped = eroded[l:r]
    colored_img = colored_img[l:r]
    # cv2.imshow("filter row:", row_clamped)
    non_black_cols = np.any(eroded > 128, axis=0) # 取全黑纵列作为分割依据

    lrs = []
    ll = -1
    for p, i in enumerate(non_black_cols):
        if i:
            if ll == -1:
                ll = p
        else:
            if ll != -1:
                lrs.append((ll, p))
                ll = -1
    p = len(non_black_cols)
    if ll != -1: # 末尾处理
        lrs.append((ll, p))
        ll = -1


    for l, r in lrs:
        res = colored_img[:, l:r]
        seg.append(res)
        # cv2.imshow(f"{l}-{r}", res)
    return seg
    # cv2.waitKey()


TESTDIR = Path(r'F:\hakanai\Assets\testcase')

if __name__ == '__main__':
    for i in os.listdir(TESTDIR):
        imgpath = TESTDIR / i # 本目录下保证都是jpg文件
        # img = cv2.imread(str(imgpath.absolute()))
        img = cv2.imread(str(imgpath.absolute()), cv2.IMREAD_GRAYSCALE)
        cimg = cv2.imread(str(imgpath.absolute()))
        segment_img(img, cimg)
