from PIL import Image
import numpy as np

import torch.utils.data as data
import torch.nn.functional as F


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def scale_shortside(img, target_width, crop_width, method=Image.BICUBIC):
    ow, oh = img.size
    shortside = min(ow, oh)
    if shortside >= target_width:
        return img
    else:
        scale = target_width / shortside
        return img.resize((round(ow * scale), round(oh * scale)), method)


def expand2square(pil_img, background_color=(0, 0, 0)):
    # Reference: https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/
    # Be robust to numpy inputs (some pipelines may pass ndarray)
    if isinstance(pil_img, np.ndarray):
        # ensure uint8 HWC
        arr = pil_img
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        pil_img = Image.fromarray(arr)

    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
