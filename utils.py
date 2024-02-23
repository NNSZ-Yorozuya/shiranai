from pathlib import Path
import os

import cv2
import numpy as np

# def resize_image(image, target_size=(45, 70)):
def resize_image(image, target_size=(45, 70)) -> np.ndarray:
    # 使用插值方法将图像缩放到目标大小
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image
    # r = resize_image(img)
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Resized Image', r)
    # cv2.waitKey(0)

def tokenize_img(imgpath: str | np.ndarray) -> np.ndarray:
    # img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(imgpath) if isinstance(imgpath, str) else imgpath
    img = resize_image(img)
    # ret, img = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY_INV)
    return img


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

def add_rotated_highlight_strip(image, offset_x_persentage, angle=150, width_persentage=0.2, intensity=0.5) -> np.ndarray:
    height, width = image.shape[:2]

    # 创建一个尺寸是原图四倍的透明图层
    overlay = np.zeros((4*height, 4*width, 3), dtype=np.uint8)

    lcenter = (2*width, 2*height) # 大画布中心
    lsize = (4*width, 4*height) # 大画布尺寸

    # 计算中心线位置
    center_line = image.shape[1] // 2
    strip_width = int(image.shape[1] * width_persentage)

    # 在这个更大的图层上绘制高光条
    # 高光条位置是在这个大画布的中心
    cv2.rectangle(overlay, (lcenter[0] - strip_width // 2, 0),
                  (lcenter[0] + strip_width // 2, lsize[1]), (255, 255, 255), -1)

    # 创建旋转矩阵，中心点是新画布的中心
    M = cv2.getRotationMatrix2D(lcenter, angle, 1)

    # 应用旋转矩阵，旋转高光条
    rotated_overlay = cv2.warpAffine(overlay, M, lsize)

    offset_x_persentage = int(offset_x_persentage * width)
    # print(offset_x_persentage)
    # 根据水平平移量（offset_x）调整裁剪的起始点
    start_x = width//2 - offset_x_persentage
    start_y = lcenter[1]
    end_x = start_x + width
    end_y = start_y + height

    # 确保裁剪坐标不会超出旋转后图层的边界
    # cv2.imshow('',rotated_overlay)
    start_x = max(0, min(start_x, lsize[0] - width))
    end_x = start_x + width
    rotated_overlay_cropped = rotated_overlay[start_y:end_y, start_x:end_x]
    # cv2.imshow(f"{start_x}",rotated_overlay_cropped)
    combined = cv2.addWeighted(image, 1, rotated_overlay_cropped, intensity, 0)

    # 返回添加了高光的图像
    return combined

def load_augumented_data_as_dict(labeled_dir: Path) -> dict:
    ds = {}
    for label in os.listdir(labeled_dir):
        for fn in os.listdir(labeled_dir / label):
            imgpath = labeled_dir / label / fn
            original = cv2.imread(str(imgpath.absolute()))
            tokenized = tokenize_img(str(imgpath.absolute()))
            augs = [original]
            # cv2.imshow('original', original)
            ext = []
            for source in augs:
                for i in range(20,21,2):
                    redded = red_augment(source, i / 100)
                    # cv2.imshow('redded-'+ str(i), redded)
                    ext.append(redded)
            augs.extend(ext)
            ext.clear()
            for source in augs:
                for ofs in range(-160, 41, 10):
                    highlight = add_rotated_highlight_strip(source, ofs / 100)
                    # cv2.imshow('highlight-'+ str(ofs), highlight)
                    ext.append(highlight)
            augs.extend(ext)
            ext.clear()
            for source in augs:
                for i in range(3, 13, 2):
                    blurred = blur_augment(source, i)
                    # cv2.imshow('blurred-'+ str(i), blurred)
                    ext.append(blurred)
            augs.extend(ext)
            tokenize_imgs = []
            for p, source in enumerate(augs):
                # cv2.imshow('source', source)
                # cv2.waitKey()
                tokenized = tokenize_img(str(imgpath.absolute()))
                # cv2.imshow('tokenized', tokenized)
                tokenize_imgs.append((tokenized, source))
                # cv2.imwrite(str((AUGUMENTATION_DIR / label / (fn + '-' + str(p) + '.png')).absolute()), source)
            # cv2.waitKey()
            ds.setdefault(label, []).extend(tokenize_imgs)
    return ds

def load_augumented_data_as_list(labeled_dir: Path) -> list:
    """炼丹用"""
    ds = []
    for label in os.listdir(labeled_dir):
        for fn in os.listdir(labeled_dir / label):
            imgpath = labeled_dir / label / fn
            original = cv2.imread(str(imgpath.absolute()))
            tokenized = tokenize_img(str(imgpath.absolute()))
            augs = [original]
            # cv2.imshow('original', original)
            ext = []
            for source in augs:
                for i in range(20,21,2):
                    redded = red_augment(source, i / 100)
                    # cv2.imshow('redded-'+ str(i), redded)
                    ext.append(redded)
            augs.extend(ext)
            ext.clear()
            for source in augs:
                for ofs in range(-160, 41, 10):
                    highlight = add_rotated_highlight_strip(source, ofs / 100)
                    # cv2.imshow('highlight-'+ str(ofs), highlight)
                    ext.append(highlight)
            augs.extend(ext)
            ext.clear()
            for source in augs:
                for i in range(3, 13, 2):
                    blurred = blur_augment(source, i)
                    # cv2.imshow('blurred-'+ str(i), blurred)
                    ext.append(blurred)
            augs.extend(ext)
            tokenize_imgs = []
            for p, source in enumerate(augs):
                # cv2.imshow('source', source)
                # cv2.waitKey()
                tokenized = tokenize_img(str(imgpath.absolute()))
                # cv2.imshow('tokenized', tokenized)
                tokenize_imgs.append((label, tokenized))
                # cv2.imwrite(str((AUGUMENTATION_DIR / label / (fn + '-' + str(p) + '.png')).absolute()), source)
            # cv2.waitKey()
            ds.extend(tokenize_imgs)
    return ds