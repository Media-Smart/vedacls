import sys
import random
import numbers
import collections

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

import numpy as np
import torchvision.transforms.functional as F
import imgaug.augmenters as iaa

from PIL import Image


try:
    import accimage
except ImportError:
    accimage = None

from .registry import PIPELINES


__all__ = ["Lambda", "Compose", "ToTensor", "Normalize",
           "RandomHorizontalFlip", "RandomVerticalFlip",
           "RandomRotation", "ColorJitter", "ResizeKeepRatio",
           "RandomEdgeShifting", "GaussianNoiseChannelWise"]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


@PIPELINES.register_module
class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


@PIPELINES.register_module
class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


@PIPELINES.register_module
class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


@PIPELINES.register_module
class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        ratio(float): probability of a image to be rotated. 0 <= ratio <= 1
        mode(bool): mode of degrees selections. 'range': degree will be randomly selected from the
            given range; 'constant': degree will be randomly selected from the given constants
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fillcolor(tuple): the color used to padding after rotation. a tuple with length of 3, (r, g, b).

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, ratio=0.5, mode='range', resample=Image.NEAREST, expand=True, center=None, fillcolor=None):
        self.mode = mode
        if self.mode == 'range':
            if isinstance(degrees, numbers.Number):
                if degrees < 0:
                    raise ValueError("'range' mode: If degrees is a single number, it must be positive.")
                self.degrees = (-degrees, degrees)
            else:
                if len(degrees) != 2:
                    raise ValueError("'range' mode: If degrees is a sequence, it must be of len 2.")
                self.degrees = degrees

        elif self.mode == 'constant':
            if isinstance(degrees, Iterable):
                self.degrees = degrees
            else:
                raise ValueError("'constant' mode: degrees must be Iterable")
        else:
            raise NotImplementedError("currently, method only supports mode 'range' and 'constant'")

        self.ratio = ratio
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fillcolor = fillcolor

    def get_params(self, degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        if self.mode == 'range':
            angle = random.uniform(degrees[0], degrees[1])
        elif self.mode == 'constant':
            angle = random.choice(self.degrees)
        else:
            angle = 0

        return angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """
        if random.random() < self.ratio:
            angle = self.get_params(self.degrees)
        else:
            angle = 0
        return img.rotate(angle, resample=self.resample, expand=self.expand,
                          center=self.center, fillcolor=self.fillcolor)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


@PIPELINES.register_module
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)

        return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


@PIPELINES.register_module
class ResizeKeepRatio(object):
    """Resize the input PIL Image to given size with ratio kept and constant padding

    Args:
        size (sequence): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            longer edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * width/height)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
        fillcolor(tuple): the color used to padding after resize.
            a tuple with length of 3, (r, g, b).

    """

    def __init__(self, size, interpolation=Image.BILINEAR, fillcolor=(0, 0, 0)):
        assert isinstance(size, int) or(isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.fillcolor = fillcolor

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return self.resize_keep_ratio(img)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

    def resize_keep_ratio(self, img):
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
        if not (isinstance(self.size, int) or (isinstance(self.size, tuple) and len(self.size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(self.size))

        h, w = img.size
        if isinstance(self.size, int):
            ratio = self.size / max(h, w)
            h_ = self.size
            w_ = self.size
        else:
            ratio = min(self.size[0]/h, self.size[1]/w)
            h_ = self.size[0]
            w_ = self.size[1]

        h_o, w_o = int(h * ratio), int(w * ratio)

        img_o = img.resize((w_o, h_o), resample=self.interpolation)
        top = (h_ - h_o) // 2
        bottom = h_ - h_o - top
        left = (w_ - w_o) // 2
        right = w_ - w_o - left

        img_out = F.pad(img_o, (left, top, right, bottom), fill=self.fillcolor, padding_mode='constant')

        return img_out


@PIPELINES.register_module
class RandomEdgeShifting(object):
    """shift edges of the given PIL Image randomly with a given probability.

    Args:
        shift_factor(int or tuple): pixel length of the edges will be shifted.
            If shift_factor is an int, it will be applied to x1, y1, x2, y2.
            If it is an tuple, it will be applied to x1, y1, x2, y2 correspondingly.
        p (float): probability of the image being processed. Default value is 0.5
    """

    def __init__(self, shift_factor, p=0.5):
        self.shift_factor = shift_factor
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be shifted.

        Returns:
            PIL Image: Randomly shifted image.
        """
        if isinstance(self.shift_factor, int):
            x1_sf = y1_sf = x2_sf = y2_sf = self.shift_factor
        elif isinstance(self.shift_factor, tuple):
            x1_sf, y1_sf, x2_sf, y2_sf = self.shift_factor
        else:
            raise TypeError('shift_factor should be int or tuple(lenth=4). Got {}'.format(type(self.shift_factor)))

        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if random.random() < self.p:
            w, h = img.size
            x1 = 0 + np.random.randint(0, x1_sf+1)
            y1 = 0 + np.random.randint(0, y1_sf+1)
            x2 = w - np.random.randint(0, x2_sf+1)
            y2 = h - np.random.randint(0, y2_sf+1)
            img = img.crop((x1, y1, x2, y2))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p) + ', edge shift_factor = {}'.format(self.shift_factor)


@PIPELINES.register_module
class GaussianNoiseChannelWise(object):
    def __init__(self,
                 ratio=0.5,
                 r=dict(loc=(-0.25, 0.25), scale=(2.5, 3.2), per_channel=True),
                 g=dict(loc=(-0.15, 0.15), scale=(1.8, 2.4), per_channel=True),
                 b=dict(loc=(-0.35, 0.35), scale=(2.4, 3), per_channel=True)):

        self.ratio = ratio
        self.r = r
        self.g = g
        self.b = b
        self.aug_r = iaa.AdditiveGaussianNoise(**r)
        self.aug_g = iaa.AdditiveGaussianNoise(**g)
        self.aug_b = iaa.AdditiveGaussianNoise(**b)

    def __call__(self, img):
        if random.random() < self.ratio:
            img = np.array(img)
            img_r = np.expand_dims(self.aug_r.augment_image(img[:, :, 0]), axis=2)
            img_g = np.expand_dims(self.aug_g.augment_image(img[:, :, 1]), axis=2)
            img_b = np.expand_dims(self.aug_b.augment_image(img[:, :, 2]), axis=2)
            img_o = np.concatenate((img_r, img_g, img_b), axis=2)
            return Image.fromarray(img_o)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(p={})'.format(self.p)
        for param in [self.r, self.g, self.b]:
            format_string += ', r={}'.format(param)
        return format_string
