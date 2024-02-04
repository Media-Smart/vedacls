import random

import cv2
import torch
import numpy as np
import albumentations as albu
from albumentations import ImageOnlyTransform, DualTransform
from albumentations.augmentations.functional import _maybe_process_in_chunks
import albumentations.augmentations.functional as F

from .registry import TRANSFORMS


@TRANSFORMS.register_module
class ToTensor(ImageOnlyTransform):
    def __init__(self):
        super(ToTensor, self).__init__(always_apply=True)

    def apply(self, img, **params):
        """
        Args:
            img(torch.Tensor): image, shape 1*C*H*W
        """
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[:, :, None]
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)
        else:
            raise TypeError('img shoud be np.ndarray. Got {}'
                            .format(type(img)))

        return img

    def get_transform_init_args_names(self):
        return ()


@TRANSFORMS.register_module
class RandomEdgeShifting(ImageOnlyTransform):
    """shift edges of the given PIL Image randomly with a given probability.

    Args:
        shift_factor(int or tuple): pixel length of the edges will be shifted.
            If shift_factor is an int, it will be applied to x1, y1, x2, y2.
            If it is an tuple, it will be applied to x1, y1, x2, y2.
        p (float): probability of the image being processed.
            Default value is 0.5
    """
    def __init__(self, shift_factor, always_apply=False, p=0.5):
        super(RandomEdgeShifting, self).__init__(always_apply, p)
        self.shift_factor = shift_factor

    def get_params(self):
        shift_factor = self.shift_factor
        if isinstance(shift_factor, int):
            shift_factor = (shift_factor,) * 4
        assert len(shift_factor) == 4

        offset = tuple(np.random.randint(0, factor+1) for factor in
                       shift_factor)

        return {'offset': offset}

    def apply(self, img, offset=(0, 0, 0, 0), **params):
        x_min = offset[0]
        y_min = offset[1]
        x_max = params['cols'] - offset[2]
        y_max = params['rows'] - offset[3]

        return img[y_min:y_max, x_min=x_max, :]

    def get_transform_init_args_names(self):
        return ('shift_factor',)


@TRANSFORMS.register_module
class ExpandRotate(ImageOnlyTransform):
    """Rotate the input by an angle selected randomly from the uniform
    distribution without crop.

    Args:
        limit ((int, int) or int): range from which a random angle is picked.
            If limit is a single int an angle is picked from (-limit, limit).
            Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the
            interpolation algorithm. Should be one of: cv2.INTER_NEAREST,
            cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
            cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel
            extrapolation method. Should be one of: cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101. Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if
            border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        always_apply=False,
        p=0.5,
    ):
        super(ExpandRotate, self).__init__(always_apply, p)
        self.limit = albu.to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value

    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        height, width = img.shape[:2]
        center = (width / 2, height / 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        abs_cos = abs(matrix[0, 0])
        abs_sin = abs(matrix[0, 1])

        bound_width = int(height * abs_sin + width * abs_cos)
        bound_height = int(height * abs_cos + width * abs_sin)

        matrix[0, 2] += bound_width / 2 - center[0]
        matrix[1, 2] += bound_height / 2 - center[1]

        warp_fn = _maybe_process_in_chunks(
            cv2.warpAffine, M=matrix, dsize=(bound_width, bound_height),
            flags=interpolation, borderMode=self.border_mode,
            borderValue=self.value)

        return warp_fn(img)

    def get_params(self):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}

    def get_transform_init_args_names(self):
        return ("limit", "interpolation", "border_mode", "value")
