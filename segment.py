from PIL import Image
from pathlib import Path
import cv2
import os
import uuid
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

    non_black_rows = np.sum(eroded > 128, axis=1)
    # 取最大连续的非全黑行
    l = 0
    r = -1
    ll = -1
    for p, i in enumerate(non_black_rows):
        if i > img.shape[1] * 0.2:
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

    # row_clamped = eroded[l:r]
    eroded = eroded[l:r]
    colored_img = colored_img[l:r]
    # cv2.imshow("filter row:", colored_img)
    # cv2.waitKey()

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

    THR = 0.10
    l_list = [] # 去掉尺寸差异大于中位数的10%的
    for p, (l, r) in enumerate(lrs):
        l_list.append(r - l)
    l_list.sort()
    print(l_list)
    l_mid = l_list[len(l_list) // 2]


    for l, r in lrs:
        if abs(r - l - l_mid) > THR * l_mid:
            continue
        res = colored_img[:, l:r]
        seg.append(res)
        # cv2.imshow(f"er{l}-{r}", eroded[:, l:r])
        # cv2.imshow(f"{l}-{r}", res)
        # cv2.waitKey()
    return seg
    # cv2.waitKey()

ROOTDIR = Path(r'Assets')
TESTDIR = ROOTDIR / 'testset'
UNLABELDIR = ROOTDIR / 'unlabel' / 'testset'

def output(segs: list[np.ndarray]):
    UNLABELDIR.mkdir(exist_ok=True, parents=True)
    for s in segs:
        # Image.fromarray(s).save(str((UNLABELDIR / (uuid.uuid4().hex + '.png')).absolute()))
        cv2.imwrite(str((UNLABELDIR / (uuid.uuid4().hex + '.png')).absolute()), s)

if __name__ == '__main__':
    for i in os.listdir(TESTDIR):
        imgpath = TESTDIR / i # 本目录下保证都是jpg文件
        # img = cv2.imread(str(imgpath.absolute()))
        img = cv2.imread(str(imgpath.absolute()), cv2.IMREAD_GRAYSCALE)
        cimg = cv2.imread(str(imgpath.absolute()))
        segs = segment_img(img, cimg)
        output(segs)

