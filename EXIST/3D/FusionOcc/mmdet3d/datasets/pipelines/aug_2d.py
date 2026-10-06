import numpy as np
import math

from PIL import Image
from PIL import ImageEnhance


class Grid(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size[1]
        w = img.size[0]

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))

        d = np.random.randint(self.d1, self.d2)
        # d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        # mask = torch.from_numpy(mask).float()
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        if self.mode == 1:
            mask = 1 - mask
        img = img * mask

        return img


def get_grid_mask_img(img, d1=2, d2=100, rotate=(1, 120), ratio=(0.05, 0.25)):
    if type(rotate) == tuple or type(rotate) == list:
        rotate = np.random.randint(rotate[0], rotate[1])
    if type(ratio) == tuple or type(ratio) == list:
        ratio = np.random.uniform(ratio[0], ratio[1])
    gd = Grid(d1, d2, rotate=rotate, ratio=ratio)
    return Image.fromarray(gd(img))


def get_darker_img(img, ratio=(0.5, 1.5)):
    brighter = ImageEnhance.Brightness(img)
    if type(ratio) == tuple or type(ratio) == list:
        ratio = np.random.uniform(ratio[0], ratio[1])
    return brighter.enhance(ratio)


def get_contrast_img(img, ratio=(0.7, 1.3)):
    contraster = ImageEnhance.Contrast(img)
    if type(ratio) == tuple or type(ratio) == list:
        ratio = np.random.uniform(ratio[0], ratio[1])
    return contraster.enhance(ratio)
