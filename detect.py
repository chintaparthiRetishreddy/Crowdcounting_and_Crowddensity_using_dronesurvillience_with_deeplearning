import os
import cv2
import random
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt 

import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transform
import misc.transforms as own_transforms

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio 
from PIL import Image, ImageOps

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True


mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()
LOG_PARA = 100.0


def generate_point_map(kpoint):
    rate = 1
    pred_coor = np.nonzero(kpoint)
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255  # 22
    # count = len(pred_coor[0])
    coord_list = []
    for i in range(0, len(pred_coor[0])):
        h = int(pred_coor[0][i] * rate)
        w = int(pred_coor[1][i] * rate)
        coord_list.append([w, h])
        cv2.circle(point_map, (w, h), 3, (0, 0, 0), -1)

    print(kpoint.shape)
    cv2.imshow("framex", point_map)
    return point_map


def test(file, model_path):
    net = CrowdCounter(cfg.GPU_ID, 'MCNN')
    net.cuda()
    net.load_state_dict(torch.load(model_path))

    gts = []
    preds = []

    record = open('submmited.txt', 'w+')

    img = Image.open(file)
    print(img.size)
    if img.mode == 'L':
        img = img.convert('RGB')
    
    img = img_transform(img)[None, :, :, :]
    
    with torch.no_grad():
        img = Variable(img).cuda()
        crop_imgs, crop_masks = [], []
        b, c, h, w = img.shape
        rh, rw = 576, 768
        for i in range(0, h, rh):
            gis, gie = max(min(h-rh, i), 0), min(h, i+rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w-rw, j), 0), min(w, j + rw)
                crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                mask = torch.zeros(b, 1, h, w).cuda()
                mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                crop_masks.append(mask)
        crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

        # Forward 

        crop_preds = []
        nz, bz = crop_imgs.size(0), 1
        for i in range(0, nz, bz):
            gs, gt = i, min(nz, i+bz)
            crop_pred = net.test_forward(crop_imgs[gs:gt])
            crop_preds.append(crop_pred)
        crop_preds = torch.cat(crop_preds, dim=0)

        # Splice them into original size
        idx = 0
        pred_map = torch.zeros(b, 1, h, w).cuda()
        for i in range(0, h, rh):
            gis, gie = max(min(h-rh, i), 0), min(h, i+rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                idx += 1

        # generate_point_map(pred_map[0][0])

        # Average for overlapping area
        mask = crop_masks.sum(dim=0).unsqueeze(0)

        pred_map_x = pred_map[0][0].cpu().data.numpy().copy()
        pred_map = pred_map/mask


        # print(np.max(), np.min(pred_map.cpu().data.numpy()))
        print(pred_map.shape)
        x = pred_map.cpu().data.numpy()*255
        x = x[0][0].reshape(3436, -1)
        cv2.imshow("ff", x)
        cv2.waitKey(0)

    pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

    pred = np.sum(pred_map) / LOG_PARA

    print(f'{file} {pred:.4f}', file=record)
    print(f'{file} {pred:.4f}')

    record.close()



if __name__ == '__main__':
    file = '0001.jpg' 
    model = 'exppp\MCNN-all_ep_907_mae_218.5_mse_700.6_nae_2.005.pth'
    test(file, model)