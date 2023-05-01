import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # NWPU root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2

import torch
from torch.autograd import Variable
import misc.transforms as own_transforms
import torchvision.transforms as standard_transforms

from misc.utils import *

from PIL import Image

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


def nwpu_count(frame, net):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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

                # Average for overlapping area
        mask = crop_masks.sum(dim=0).unsqueeze(0)

        pred_map = pred_map/mask

    pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

    pred = np.sum(pred_map) / LOG_PARA

    return pred



if __name__ == "__main__":
    frame = cv2.imread('./0001.jpg')
    nwpu_count(frame)