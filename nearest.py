from pathlib import Path
import cv2
import os
import uuid
import numpy as np
import datetime
import shutil
from utils import tokenize_img, blur_augment, red_augment, add_rotated_highlight_strip

ROOTDIR = Path(r'D:\SETU\shiranai\Assets')
LABEL = ROOTDIR / 'label'
AUGUMENTATION_DIR = ROOTDIR / 'aug'

OUTDIR = ROOTDIR / 'nearest_output'
INPUTDIR = ROOTDIR / 'segmented'

def output(feature: np.ndarray, label: str, filename: str = None):
    OUTDIR.mkdir(exist_ok=True,parents=True)
    OUTLABELDIR = OUTDIR / label
    OUTLABELDIR.mkdir(exist_ok=True,parents=True)
    cv2.imwrite(str((OUTLABELDIR / (filename or (uuid.uuid4().hex + '.png'))).absolute()), feature)



if __name__ == '__main__':
    AUGUMENTATION_DIR.mkdir(exist_ok=True, parents=True)
    ds = {} # 训练样本空间
    now = datetime.datetime.now()
    for label in os.listdir(LABEL):
        (AUGUMENTATION_DIR / label).mkdir(exist_ok=True, parents=True)
        for fn in os.listdir(LABEL / label):
            imgpath = LABEL / label / fn
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
                cv2.imwrite(str((AUGUMENTATION_DIR / label / (fn + '-' + str(p) + '.png')).absolute()), source)
            # cv2.waitKey()
            ds.setdefault(label, []).extend(tokenize_imgs)
    print('train set loaded in', datetime.datetime.now() - now)
    shutil.rmtree(OUTDIR)
    for sample in os.listdir(INPUTDIR):
        imgpath = INPUTDIR / sample
        cimg = cv2.imread(str(imgpath.absolute()))
        img = tokenize_img(str(imgpath.absolute()))
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