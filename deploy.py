# 部署，用于单张图片识别
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from utils import tokenize_img
from data import classes_int2str

SOURCE = Path('Assets/segmented/screenshot')
OUTDIR = Path('Assets/predicted/cnn2')
OUTDIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval_model(
    model: nn.Module,
    image: torch.Tensor,
):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted[0]

def eval(fn: str | np.ndarray):
    from nets import SimpleClassifier, CNN2
    model = CNN2()
    sd = torch.load(os.path.join('trained', 'cnn2-adam-0.001-0.9-30-0.1-8.ckpt'))
    model.load_state_dict(sd)
    model = model.to(device)

    tokenized = tokenize_img(fn)
    
    pd = eval_model(model, torch.from_numpy(tokenized).permute(2, 0, 1).float().unsqueeze(0))
    
    label = pd.item()
    label = classes_int2str[label]

    # print(label, fn)
    return label

if __name__ == "__main__":
    for fn in os.listdir(SOURCE.absolute()):
        if fn.endswith('.jpg') or fn.endswith('.png'):
            label = eval(str((SOURCE / fn).absolute()))
            labeldir = OUTDIR / label
            labeldir.mkdir(parents=True, exist_ok=True)
            shutil.copy(fn, labeldir.absolute())