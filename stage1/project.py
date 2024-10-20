import cv2
import numpy as np
from skimage import exposure


def equalize(img):
    p2, p98 = np.percentile(img, (0.5, 99.5))
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0., 1.))
    img = np.clip(img, 0, 1.)
    # img = img.astype(np.int16)
    # Equalization
    # img_eq = exposure.equalize_hist(img)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.01)
    img_adapteq = (img_adapteq * 255).astype(np.uint8)
    return img_adapteq


# P123 noise
if __name__ == '__main__':
    import json
    import os
    import nrrd

    data_dir = ''

    for case in sorted(os.listdir(data_dir)):
        nrrd_path = os.path.join(data_dir, case)
        hu_img, img_opts = nrrd.read(nrrd_path)

        for i in [0, 1, 2]:
            stack_max = hu_img.max(axis=i)

            eq_max = equalize(stack_max)
            cv2.imshow('max', eq_max)
            cv2.waitKey()

