from PIL import Image
from pathlib import Path
import cv2
import os
import uuid
import numpy as np
import hashlib

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

def laplacian_sharpening(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]],
    dtype=np.float32)

    image = cv2.filter2D(image, -1, kernel)
    image[image < 0] = 0
    image[image > 255] = 255
    return image

def constuct_bounding_box(img, ref_img):
    height, width = img.shape[:2]
    erode_scale = max(2, int(max(height, width) * 0.0021)) # 4 / 1920
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # bounding_boxes = []

    boxes_img = np.zeros((height, width), dtype=np.uint8)
    test_canvas = np.zeros((height, width), dtype=np.uint8)
    mg = erode_scale * 2 # 观察到每张牌比较方正，搞一个边缘框，测试落在其边界的白色像素数量
    m1 = max(1, mg // 3)
    m2 = m1 + 1
    m3 = 1
    # print(mg, m1, m2)
    for contour in contours:
        # 计算每个轮廓的包围盒
        x, y, w, h = cv2.boundingRect(contour)

        test_canvas[y+m1:y+h-m1, x+m1:x+w-m1] |= 127
        test_canvas[y+m1:y+h-m1, x+m1:x+w-m1] |= ref_img[y+m1:y+h-m1, x+m1:x+w-m1]
        test_canvas[y+m2:y+h-m2, x+m2:x+w-m2] &= 0

        if w / h > 0.66 or w / h < 0.6:
            continue
        x+=m3
        y+=m3
        w-= 2 * m3
        h-= 2 * m3
        white = cv2.countNonZero(ref_img[y+m1:y+h-m1, x+m1:x+w-m1])
        siz = (h - 2 * m1) * (w - 2 * m1)
        if white < siz * 0.47:
            # print('skip', x, y, x + w, y + h, white / siz if siz > 0 else 0)
            # cv2.rectangle(boxes_img, (x, y), (x + w, y + h), 128, -1)
            continue
        inner_white = cv2.countNonZero(ref_img[y+m2:y+h-m2, x+m2:x+w-m2])
        outer_white = white - inner_white
        inner_size = (h - m2 * 2) * (w - m2 * 2)
        outer_size = siz - inner_size

        perimeter = cv2.arcLength(contour, True)
        if perimeter < 100:
            continue

        # print(x, y, x + w, y + h, outer_white / outer_size if outer_size > 0 else 0, 'perimeter:', perimeter)
        if outer_white < outer_size * 0.90: # 大方块减去小方块，得到边缘框
            continue
        cv2.rectangle(boxes_img, (x, y), (x + w, y + h), 255, -1)
        # cv2.rectangle(boxes_img, (x+erode_scale, y+erode_scale), (x+w-erode_scale, y+h-erode_scale), 128, -1)
    # cv2.imshow("test_img", test_canvas)
    return boxes_img


def segment_img(img, colored_img):
    seg = []
    height, width = colored_img.shape[:2]

    # cannied = cv2.Canny(img, 10, 10, apertureSize=3, L2gradient=True)
    # cv2.imshow("cannied:", cannied)

    # img = img * (1 - cannied // 255)
    # cv2.imshow("Original", img)
    _, original_thresh = cv2.threshold(img, 144, 200, cv2.THRESH_BINARY)
    
    sharped = laplacian_sharpening(img)
    sharped = laplacian_sharpening(sharped) # 搞两次锐化

    _, thresh = cv2.threshold(sharped, 144, 200, cv2.THRESH_BINARY)

    # cv2.imshow("thresh", thresh)
    # cv2.imshow("original_thresh", original_thresh)
    erode_scale = max(2, int(max(height, width) * 0.0021)) # 4 / 1920
    # print('erosion scale:', erode_scale)
    kernel = np.ones((erode_scale,erode_scale),dtype="uint8") # 查以某点为中心 4*4 区域内是不是全是这个颜色。这个超参根据需要调整

    eroded = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel=kernel)
    original_eroded = cv2.morphologyEx(original_thresh, cv2.MORPH_OPEN, kernel=kernel)
    # cv2.imshow("eroded", eroded)
    # cv2.waitKey()
    boxes_img = constuct_bounding_box(eroded, original_thresh) | constuct_bounding_box(original_eroded, original_thresh)
    
    # cv2.imshow("boxes_img before:", boxes_img)

    dilate_scale = int(max(height, width) * 0.00261)
    dilated = cv2.dilate(
        boxes_img,
        cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_scale, dilate_scale)),
        iterations=1
    ) # 膨胀5像素

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dilated_boxes = np.zeros((height, width), dtype=np.uint8)
    for contour in contours:
        # 计算每个轮廓的包围盒
        x, y, w, h = cv2.boundingRect(contour)
        white = cv2.countNonZero(boxes_img[y:y+h, x:x+w])
        if white < w * h * 0.9:
            continue
        cv2.rectangle(dilated_boxes, (x, y), (x + w, y + h), 255, -1)

    # cv2.imshow("dilated_boxes:", dilated_boxes)
    # cv2.imshow("boxes_img:", boxes_img)

    # cv2.waitKey()
    
    # black_image = np.zeros((height, width, 3), dtype=np.uint8)
    # cv2.drawContours(black_image, contours, -1, (0, 0, 255), 3)
    # cv2.imshow("colored_img contours:", black_image)
    # cv2.imshow("contours AABB:", boxes_img)
    boxes_img = boxes_img & dilated_boxes

    non_black_rows = np.sum(boxes_img > 128, axis=1)
    # print(len(non_black_rows))
    # 取最大连续的非全黑行
    l = 0
    r = -1
    ll = -1
    for p, i in enumerate(non_black_rows):
        if i > img.shape[1] * 0.1:
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
    dilated_boxes = dilated_boxes[l:r]
    boxes_img = boxes_img[l:r]

    # cv2.imshow("filter row:", eroded)
    # cv2.waitKey()

    non_black_cols = np.any(boxes_img > 128, axis=0) # 取全黑纵列作为分割依据

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
INPUTDIR = ROOTDIR / 'raw' / 'screenshot'
OUTDIR = ROOTDIR / 'segmented' / 'screenshot'

def output(segs: list[np.ndarray], src_name: str):
    OUTDIR.mkdir(exist_ok=True, parents=True)
    for i, s in enumerate(segs):
        # Image.fromarray(s).save(str((OUTDIR / (uuid.uuid4().hex + '.png')).absolute()))
        cv2.imwrite(str((OUTDIR /  (
            f'{src_name}-{i}-' + 
            hashlib.md5(s.astype('uint8')).hexdigest()[:6] + 
        '.png')).absolute()), s)

if __name__ == '__main__':
        # i = '2.jpg'
        # i = 'Snipaste_2024-02-20_01-05-53.png'
    for i in os.listdir(INPUTDIR):
        imgpath = INPUTDIR / i
        if imgpath.is_dir():
            continue
        # img = cv2.imread(str(imgpath.absolute()))
        print(i)
        img = cv2.imread(str(imgpath.absolute()), cv2.IMREAD_GRAYSCALE)
        cimg = cv2.imread(str(imgpath.absolute()))
        segs = segment_img(img, cimg)
        output(segs, i)

