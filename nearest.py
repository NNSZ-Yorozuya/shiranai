from typing import Tuple
from PIL import Image
from pathlib import Path
import cv2
import os
import uuid
import numpy as np
import datetime
import shutil

# def resize_image(image, target_size=(45, 70)):
def resize_image(image, target_size=(27, 42)):
    # 使用插值方法将图像缩放到目标大小
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image
    # r = resize_image(img)
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Resized Image', r)
    # cv2.waitKey(0)

ROOTDIR = Path(r'D:\SETU\shiranai\Assets')
LABEL = ROOTDIR / 'label'
OUTDIR = ROOTDIR / 'nearest'

INPUTDIR = ROOTDIR / 'unlabel' / 'testset'

def output(feature: np.ndarray, label: str, filename: str = None):
    OUTDIR.mkdir(exist_ok=True,parents=True)
    OUTLABELDIR = OUTDIR / label
    OUTLABELDIR.mkdir(exist_ok=True,parents=True)
    cv2.imwrite(str((OUTLABELDIR / (filename or (uuid.uuid4().hex + '.png'))).absolute()), feature)

def tokenize_img(imgpath: str) -> Tuple[np.ndarray, np.ndarray]:
    cimg = cv2.imread(imgpath)
    # img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(imgpath)
    img = resize_image(img)
    # ret, img = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY_INV)
    return img, cimg

def blur_augment(img: np.ndarray, k=3) -> np.ndarray:
    # 高斯模糊
    return cv2.GaussianBlur(img, (k, k), 0)

def red_augment(img: np.ndarray, w=0.2) -> np.ndarray:
    # 创建一个红色的滤镜，数值中红色分量被设置得较高
    red_filter = np.full_like(img, (0, 0, 255))

    # 通过加权合并原图与红色滤镜来给图像加上红色滤镜
    # cv2.addWeighted参数：(第一个图像, 第一个图像的权重, 第二个图像, 第二个图像的权重, gamma值)
    # 调整权重来控制红色滤镜的强度
    red_image = cv2.addWeighted(img, 1.-w, red_filter, w, 0)

    return red_image

def add_rotated_highlight_strip(image, strip_width, angle, intensity=0.5):
    # 创建一个与原图大小相同的透明图层
    overlay = np.zeros_like(image, dtype=np.uint8)

    # 计算中心线位置
    center_line = image.shape[1] // 2

    # 在透明图层上绘制矩形条
    cv2.rectangle(overlay, (center_line - strip_width // 2, 0),
                  (center_line + strip_width // 2, image.shape[0]), (255, 255, 255), -1)

    # 创建一个旋转矩阵，用于旋转高光条
    M = cv2.getRotationMatrix2D((center_line, image.shape[0] // 2), angle, 1)

    # 应用旋转矩阵，旋转高光条
    rotated_overlay = cv2.warpAffine(overlay, M, (image.shape[1], image.shape[0]))

    # 将旋转后的图层合并到原始图像上，使用叠加的方式
    combined = cv2.addWeighted(image, 1, rotated_overlay, intensity, 0)

    # 返回添加了高光的图像
    return combined


if __name__ == '__main__':
    ds = {} # 训练样本空间
    now = datetime.datetime.now()
    for label in os.listdir(LABEL):
        for i in os.listdir(LABEL / label):
            imgpath = LABEL / label / i
            tokenized, original = tokenize_img(str(imgpath.absolute()))
            cv2.imshow('original', original)
            add_rotated_highlight_strip(original, 10, 45)
            # for i in range(16,25,2):
            #     redded = red_augment(original, i / 100)
            #     cv2.imshow('redded-'+ str(i), redded)
            # for i in range(3, 13, 2):
            #     blurred = blur_augment(original, i)
            #     cv2.imshow('blurred-'+ str(i), blurred)
            cv2.waitKey()
            ds.setdefault(label, []).append((tokenized, original))
    print('train set loaded in', datetime.datetime.now() - now)
    shutil.rmtree(OUTDIR)
    for sample in os.listdir(INPUTDIR):
        imgpath = INPUTDIR / sample
        img, cimg = tokenize_img(str(imgpath.absolute()))
        min_delta = img
        min_dist = 9e18
        min_label = ''
        min_labeled = None
        for label, sample_set in ds.items():
            for i, ori in sample_set:
                # 近邻算法
                delta = np.abs(i.astype(np.int16) - img.astype(np.int16))
                dist = np.sum(delta)
                if dist < min_dist:
                    min_dist = dist
                    min_label = label
                    min_delta = delta
                    min_labeled = ori
        print(sample, min_label, min_dist)
        # cv2.imshow('min_label', min_delta)
        # cv2.imshow('source', cimg)
        # cv2.imshow('min_labeled', min_labeled)
        # cv2.waitKey()
        output(cimg, min_label, sample)