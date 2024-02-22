# 部署，用于单张图片识别
import torch
import os
import torch.nn as nn
import cv2
from pathlib import Path

from utils import tokenize_img

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

def eval(fn: str):
    from data import get_test_loader
    from nets import SimpleClassifier, CNN2
    model = CNN2()
    sd = torch.load(os.path.join('trained', 'cnn2-adam-0.001-0.9-30-0.1-8.ckpt'))
    model.load_state_dict(sd)
    model = model.to(device)
    
    pd = eval_model(model, torch.from_numpy(tokenize_img(fn)).permute(2, 0, 1).float().unsqueeze(0))
    print(fn, pd)

SOURCE = Path('Assets/segmented/screenshot')
if __name__ == "__main__":
    for fn in os.listdir(SOURCE.absolute()):
        if fn.endswith('.jpg') or fn.endswith('.png'):
            eval(str((SOURCE / fn).absolute()))