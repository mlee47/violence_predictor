from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import apply_affine_transform, apply_brightness_shift
import os
import glob
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Flatten, Dropout, ConvLSTM2D, BatchNormalization
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import LeakyReLU, Dense, MaxPooling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export
from skimage import segmentation, measure
import numpy as np
import random
import scipy
import PIL
import cv2
import math

### HOW TO RUN
# python featureMapVisualization.py --weights WEIGHTS_PATH --video INPUT_VIDEO_PATH

"""
###########################################################################################################
videoAugmentator.py
###########################################################################################################
"""

class GaussianBlur(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, clip):

        if isinstance(clip[0], np.ndarray):
            return [scipy.ndimage.gaussian_filter(img, sigma=self.sigma, order=0) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.filter(PIL.ImageFilter.GaussianBlur(radius=self.sigma)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))


class ElasticTransformation(object):
    def __init__(self, alpha=0, sigma=0, order=3, cval=0, mode="constant",
                 name=None, deterministic=False):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.cval = cval
        self.mode = mode

    def __call__(self, clip):

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        result = []
        nb_images = len(clip)
        for i in range(nb_images):
            image = clip[i]
            image_first_channel = np.squeeze(image[..., 0])
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=self.alpha, sigma=self.sigma)
            result.append(self._map_coordinates(
                clip[i],
                indices_x,
                indices_y,
                order=self.order,
                cval=self.cval,
                mode=self.mode))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in result]
        else:
            return result

    def _generate_indices(self, shape, alpha, sigma):
        assert (len(shape) == 2), "shape: Should be of size 2!"
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        assert (len(image.shape) == 3), "image.shape: Should be of size 3!"
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in range(image.shape[2]):
            remapped_flat = scipy.ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result


class PiecewiseAffineTransform(object):
    """
    Augmenter that places a regular grid of points on an image and randomly
    moves the neighbourhood of these point around via affine transformations.
     Args:
         displacement (init): gives distorted image depending on the valuse of displacement_magnification and displacement_kernel
         displacement_kernel (init): gives the blury effect
         displacement_magnification (float): it magnify the image
    """

    def __init__(self, displacement=0, displacement_kernel=0, displacement_magnification=0):
        self.displacement = displacement
        self.displacement_kernel = displacement_kernel
        self.displacement_magnification = displacement_magnification

    def __call__(self, clip):

        ret_img_group = clip
        if isinstance(clip[0], np.ndarray):
            im_size = clip[0].shape
            image_w, image_h = im_size[1], im_size[0]
        elif isinstance(clip[0], PIL.Image.Image):
            im_size = clip[0].size
            image_w, image_h = im_size[0], im_size[1]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        displacement_map = np.random.rand(image_h, image_w, 2) * 2 * self.displacement - self.displacement
        displacement_map = cv2.GaussianBlur(displacement_map, None,
                                            self.displacement_kernel)
        displacement_map *= self.displacement_magnification * self.displacement_kernel
        displacement_map = np.floor(displacement_map).astype('int32')

        displacement_map_rows = displacement_map[..., 0] + np.tile(np.arange(image_h), (image_w, 1)).T.astype('int32')
        displacement_map_rows = np.clip(displacement_map_rows, 0, image_h - 1)

        displacement_map_cols = displacement_map[..., 1] + np.tile(np.arange(image_w), (image_h, 1)).astype('int32')
        displacement_map_cols = np.clip(displacement_map_cols, 0, image_w - 1)

        if isinstance(clip[0], np.ndarray):
            return [img[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(img.shape) for img
                    in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [PIL.Image.fromarray(
                np.asarray(img)[(displacement_map_rows.flatten(), displacement_map_cols.flatten())].reshape(
                    np.asarray(img).shape)) for img in clip]


class Superpixel(object):
    """
    Completely or partially transform images to their superpixel representation.
    Args:
        p_replace (int) : Defines the probability of any superpixel area being
        replaced by the superpixel.
        n_segments (int): Target number of superpixels to generate.
        Lower numbers are faster.
        interpolation (str): Interpolation to use. Can be one of 'nearest',
        'bilinear' defaults to nearest
    """

    def __init__(self, p_replace=0, n_segments=0, max_size=360,
                 interpolation="bilinear"):
        self.p_replace = p_replace
        self.n_segments = n_segments
        self.interpolation = interpolation

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]
        # TODO this results in an error when n_segments is 0
        replace_samples = np.tile(np.array([self.p_replace]), self.n_segments)
        avg_image = np.mean(clip, axis=0)
        segments = segmentation.slic(avg_image, n_segments=self.n_segments,
                                     compactness=10)
        if not np.max(replace_samples) == 0:
            clip = [self._apply_segmentation(img, replace_samples, segments) for img in clip]
        if is_PIL:
            return [PIL.Image.fromarray(img) for img in clip]
        else:
            return clip

    def _apply_segmentation(self, image, replace_samples, segments):
        nb_channels = image.shape[2]
        image_sp = np.copy(image)
        for c in range(nb_channels):
            # segments+1 here because otherwise regionprops always misses
            # the last label
            regions = measure.regionprops(segments + 1,
                                          intensity_image=image[..., c])
            for ridx, region in enumerate(regions):
                # with mod here, because slic can sometimes create more
                # superpixel than requested. replace_samples then does
                # not have enough values, so we just start over with the
                # first one again.
                if replace_samples[ridx % len(replace_samples)] == 1:
                    mean_intensity = region.mean_intensity
                    image_sp_c = image_sp[..., c]
                    image_sp_c[segments == ridx] = mean_intensity

        return image_sp


class DynamicCrop(object):
    """
    Crops the spatial area of a video containing most movemnets
    """

    def __init__(self):
        pass

    def normalize(self, pdf):
        mn = np.min(pdf)
        mx = np.max(pdf)
        pdf = (pdf - mn) / (mx - mn)
        sm = np.sum(pdf)
        return pdf / sm

    def __call__(self, video, opt_flows):
        if not isinstance(video, np.ndarray):
            video = np.array(video, dtype=np.float32)
            opt_flows = np.array(opt_flows, dtype=np.float32)

        magnitude = np.sum(opt_flows, axis=0)
        magnitude = np.sum(magnitude, axis=-1)
        thresh = np.mean(magnitude)
        magnitude[magnitude < thresh] = 0
        # calculate center of gravity of magnitude map and adding 0.001 to avoid empty value
        x_pdf = np.sum(magnitude, axis=1) + 0.001
        y_pdf = np.sum(magnitude, axis=0) + 0.001
        # normalize PDF of x and y so that the sum of probs = 1
        x_pdf = x_pdf[112:208]
        y_pdf = y_pdf[112:208]
        x_pdf = self.normalize(x_pdf)
        y_pdf = self.normalize(y_pdf)
        # randomly choose some candidates for x and y
        x_points = np.random.choice(a=np.arange(
            112, 208), size=5, replace=True, p=x_pdf)
        y_points = np.random.choice(a=np.arange(
            112, 208), size=5, replace=True, p=y_pdf)
        # get the mean of x and y coordinates for better robustness
        x = int(np.mean(x_points))
        y = int(np.mean(y_points))
        video = video[:, x - 112:x + 112, y - 112:y + 112, :]
        opt_flows = opt_flows[:, x - 112:x + 112, y - 112:y + 112, :]
        # get cropped video
        return video, opt_flows

    # _____________________________________


class Add(object):
    """
    Add a value to all pixel intesities in an video.
    Args:
        value (int): The value to be added to pixel intesities.
    """

    def __init__(self, value=0):
        if value > 255 or value < -255:
            raise TypeError('The video is blacked or whitened out since ' +
                            'value > 255 or value < -255.')
        self.value = value

    def __call__(self, clip):

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.int32)
            image += self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Multiply(object):
    """
    Multiply all pixel intensities with given value.
    This augmenter can be used to make images lighter or darker.
    Args:
        value (float): The value with which to multiply the pixel intensities
        of video.
    """

    def __init__(self, value=1.0):
        if value < 0.0:
            raise TypeError('The video is blacked out since for value < 0.0')
        self.value = value

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.float64)
            image *= self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Pepper(object):
    """
    Augmenter that sets a certain fraction of pixel intensities to 0, hence
    they become black.
    Args:
        ratio (int): Determines number of black pixels on each frame of video.
        Smaller the ratio, higher the number of black pixels.
    """

    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 0, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Salt(object):
    """
    Augmenter that sets a certain fraction of pixel intesities to 255, hence
    they become white.
    Args:
        ratio (int): Determines number of white pixels on each frame of video.
        Smaller the ratio, higher the number of white pixels.
   """

    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 255, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


# -------------------------------------------

# Temporal Transformations

class TemporalBeginCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        out = clip[:self.size]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class TemporalCenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        center_index = len(clip) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(clip))

        out = clip[begin_index:end_index]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class TemporalRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        rand_end = max(0, len(clip) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(clip))

        out = clip[begin_index:end_index]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class InverseOrder(object):
    """
    Inverts the order of clip frames.
    """

    def __call__(self, clip):
        nb_images = len(clip)
        return [clip[img] for img in reversed(range(0, nb_images))]


class Downsample(object):
    def __init__(self, ratio=1.0):
        if ratio < 0.0 or ratio > 1.0:
            raise TypeError('ratio should be in [0.0 <= ratio <= 1.0]. ' +
                            'Please use upsampling for ratio > 1.0')
        self.ratio = ratio

    def __call__(self, clip):
        nb_return_frame = int(np.floor(self.ratio * len(clip)))
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=nb_return_frame)]

        return [clip[i - 1] for i in return_ind]


class Upsample(object):
    def __init__(self, ratio=1.0):
        if ratio < 1.0:
            raise TypeError('ratio should be 1.0 < ratio. ' +
                            'Please use downsampling for ratio <= 1.0')
        self.ratio = ratio

    def __call__(self, clip):
        nb_return_frame = int(np.floor(self.ratio * len(clip)))
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=nb_return_frame)]

        return [clip[i - 1] for i in return_ind]

class TemporalElasticTransformation(object):
    def __call__(self, clip):
        nb_images = len(clip)
        new_indices = self._get_distorted_indices(nb_images)
        return [clip[i] for i in new_indices]

    def _get_distorted_indices(self, nb_images):
        inverse = random.randint(0, 1)

        if inverse:
            scale = random.random()
            scale *= 0.21
            scale += 0.6
        else:
            scale = random.random()
            scale *= 0.6
            scale += 0.8

        frames_per_clip = nb_images

        indices = np.linspace(-scale, scale, frames_per_clip).tolist()
        if inverse:
            values = [math.atanh(x) for x in indices]
        else:
            values = [math.tanh(x) for x in indices]

        values = [x / values[-1] for x in values]
        values = [int(round(((x + 1) / 2) * (frames_per_clip - 1), 0)) for x in values]
        return values




"""
###########################################################################################################
dataGenrator.py
###########################################################################################################
"""

class DataGenerator(Sequence):
    def __init__(self, X_path, batch_size=1, shuffle=False, data_augmentation=True, one_hot=False, target_frames=32,
                 sample=False, normalize_=True, background_suppress=True, resize=224, frame_diff_interval=1,
                 dataset=None, mode="both"):
        # Initialize the params
        self.dataset = dataset
        self.batch_size = batch_size
        self.X_path = X_path
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        self.one_hot = one_hot
        self.target_frames = target_frames
        self.sample = sample
        self.background_suppress = background_suppress
        self.mode = mode  # ["only_frames","only_differences", "both"]
        self.resize = resize
        self.frame_diff_interval = frame_diff_interval
        self.normalize_ = normalize_
        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.X_path, self.Y_dict = self.search_data()
        # Print basic statistics information
        self.print_stats()
        return None

    def search_data(self):
        X_path = []
        Y_dict = {}

        X_path.append(self.X_path)
        Y_dict[self.X_path] = 0
        return X_path, Y_dict

    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)
        return None

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index *
                                    self.batch_size:(index + 1) * self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        # loading X
        batch_data = []
        batch_diff_data = []
        if self.mode == "both":
            for x in batch_path:
                data, diff_data = self.load_data(x)
                batch_data.append(data)
                batch_diff_data.append(diff_data)
            batch_data = np.array(batch_data)
            batch_diff_data = np.array(batch_diff_data)
        elif self.mode == "only_frames":
            for x in batch_path:
                data = self.load_data(x)
                batch_data.append(data)
            batch_data = np.array(batch_data)
        elif self.mode == "only_differences":
            for x in batch_path:
                diff_data = self.load_data(x)
                batch_diff_data.append(diff_data)
            batch_diff_data = np.array(batch_diff_data)
            # loading Y
        batch_y = [self.Y_dict[x] for x in batch_path]
        batch_y = np.array(batch_y)
        if self.mode == "both":
            return [batch_data, batch_diff_data], batch_y
        if self.mode == "only_frames":
            return [batch_data], batch_y
        if self.mode == "only_differences":
            return [batch_diff_data], batch_y

    def normalize(self, data):
        data = (data / 255.0).astype(np.float32)
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video

    def uniform_sampling(self, video, target_frames=20):
        # get total frames of input video and calculate sampling interval
        len_frames = video.shape[0]
        interval = int(np.ceil(len_frames / target_frames))
        # init empty list for sampled video and
        sampled_video = []
        for i in range(0, len_frames, interval):
            sampled_video.append(video[i])
        # calculate numer of padded frames and fix it
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad > 0:
            for i in range(-num_pad, 0):
                try:
                    padding.append(video[i])
                except:
                    padding.append(video[0])
            sampled_video += padding
        # get sampled video
        return np.array(sampled_video, dtype=np.float32)

    def random_clip(self, video, target_frames=20):
        start_point = np.random.randint(len(video) - target_frames)
        return video[start_point:start_point + target_frames]

    def color_jitter(self, video, prob=1):
        # range of s-component: 0-1
        # range of v component: 0-255
        s = np.random.rand()
        if s > prob:
            return video
        s_jitter = np.random.uniform(-0.3, 0.3)  # (-0.2,0.2)
        v_jitter = np.random.uniform(-40, 40)  # (-30,30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[..., 1] + s_jitter
            v = hsv[..., 2] + v_jitter
            s[s < 0] = 0
            s[s > 1] = 1
            v[v < 0] = 0
            v[v > 255] = 255
            hsv[..., 1] = s
            hsv[..., 2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video

    def crop_center(self, video, x_crop=10, y_crop=30):
        frame_size = np.size(video, axis=1)
        x = frame_size
        y = frame_size
        x_start = x_crop
        x_end = x - x_crop
        y_start = y_crop
        y_end = y - y_crop
        video = video[:, y_start:y_end, x_start:x_end, :]
        return video

    def random_shear(self, video, intensity, prob=0.5, row_axis=0, col_axis=1, channel_axis=2,
                     fill_mode='nearest', cval=0., interpolation_order=1):
        s = np.random.rand()
        if s > prob:
            return video
        shear = np.random.uniform(-intensity, intensity)

        for i in range(video.shape[0]):
            x = apply_affine_transform(video[i, :, :, :], shear=shear, channel_axis=channel_axis,
                                       fill_mode=fill_mode, cval=cval,
                                       order=interpolation_order)
            video[i] = x
        return video

    def random_shift(self, video, wrg, hrg, prob=0.5, row_axis=0, col_axis=1, channel_axis=2,
                     fill_mode='nearest', cval=0., interpolation_order=1):
        s = np.random.rand()
        if s > prob:
            return video
        h, w = video.shape[1], video.shape[2]
        tx = np.random.uniform(-hrg, hrg) * h
        ty = np.random.uniform(-wrg, wrg) * w

        for i in range(video.shape[0]):
            x = apply_affine_transform(video[i, :, :, :], tx=tx, ty=ty, channel_axis=channel_axis,
                                       fill_mode=fill_mode, cval=cval,
                                       order=interpolation_order)
            video[i] = x
        return video

    def random_rotation(self, video, rg, prob=0.5, row_axis=0, col_axis=1, channel_axis=2,
                        fill_mode='nearest', cval=0., interpolation_order=1):
        s = np.random.rand()
        if s > prob:
            return video
        theta = np.random.uniform(-rg, rg)
        for i in range(np.shape(video)[0]):
            x = apply_affine_transform(video[i, :, :, :], theta=theta, channel_axis=channel_axis,
                                       fill_mode=fill_mode, cval=cval,
                                       order=interpolation_order)
            video[i] = x
        return video

    def random_brightness(self, video, brightness_range):
        if len(brightness_range) != 2:
            raise ValueError(
                '`brightness_range should be tuple or list of two floats. '
                'Received: %s' % (brightness_range,))
        u = np.random.uniform(brightness_range[0], brightness_range[1])
        for i in range(np.shape(video)[0]):
            x = apply_brightness_shift(video[i, :, :, :], u)
            video[i] = x
        return video

    def gaussian_blur(self, video, prob=0.5, low=1, high=2):
        s = np.random.rand()
        if s > prob:
            return video
        sigma = np.random.rand() * (high - low) + low
        return GaussianBlur(sigma=sigma)(video)

    def elastic_transformation(self, video, prob=0.5, alpha=0):
        s = np.random.rand()
        if s > prob:
            return video
        return ElasticTransformation(alpha=alpha)(video)

    def piecewise_affine_transform(self, video, prob=0.5, displacement=3, displacement_kernel=3,
                                   displacement_magnification=2):
        s = np.random.rand()
        if s > prob:
            return video
        return PiecewiseAffineTransform(displacement=displacement, displacement_kernel=displacement_kernel,
                                        displacement_magnification=displacement_magnification)(video)

    def superpixel(self, video, prob=0.5, p_replace=0, n_segments=0):
        s = np.random.rand()
        if s > prob:
            return video
        return Superpixel(p_replace=p_replace, n_segments=n_segments)(video)

    def resize_frames(self, video):
        video = np.array(video, dtype=np.float32)
        if (video.shape[1] == self.resize and video.shape[2] == self.resize):
            return video
        resized = []
        for i in range(video.shape[0]):
            x = cv2.resize(
                video[i], (self.resize, self.resize)).astype(np.float32)
            resized.append(x)
        return np.array(resized, dtype=np.float32)

    def dynamic_crop(self, video, opt_flows):
        return DynamicCrop()(video, opt_flows)

    def random_crop(self, video, prob=0.5):
        s = np.random.rand()
        if s > prob:
            return self.resize_frames(video)
        # gives back a randomly cropped 224 X 224 from a video with frames 320 x 320
        if self.dataset == 'rwf2000' or self.dataset == 'surv':
            x = np.random.choice(
                a=np.arange(112, 320 - 112), replace=True)
            y = np.random.choice(
                a=np.arange(112, 320 - 112), replace=True)
            video = video[:, x - 112:x + 112, y - 112:y + 112, :]
        else:
            x = np.random.choice(
                a=np.arange(80, 224 - 80), replace=True)
            y = np.random.choice(
                a=np.arange(80, 224 - 80), replace=True)
            video = video[:, x - 80:x + 80, y - 80:y + 80, :]
            video = self.resize_frames(video)
        return video

    def background_suppression(self, data):
        video = np.array(data, dtype=np.float32)
        avgBack = np.mean(video, axis=0)
        video = np.abs(video - avgBack)
        return video

    def frame_difference(self, video):
        num_frames = len(video)
        k = self.frame_diff_interval
        out = []
        for i in range(num_frames - k):
            out.append(video[i + k] - video[i])
        return np.array(out, dtype=np.float32)

    def pepper(self, video, prob=0.5, ratio=100):
        s = np.random.rand()
        if s > prob:
            return video
        return Pepper(ratio=ratio)(video)

    def salt(self, video, prob=0.5, ratio=100):
        s = np.random.rand()
        if s > prob:
            return video
        return Salt(ratio=ratio)(video)

    def inverse_order(self, video, prob=0.5):
        s = np.random.rand()
        if s > prob:
            return video
        return InverseOrder()(video)

    def downsample(self, video):
        video = Downsample(ratio=0.5)(video)
        return np.concatenate((video, video), axis=0)

    def upsample(self, video):
        num_frames = len(video)
        video = Upsample(ratio=2)(video)
        s = np.random.randint(0, 1)
        if s:
            return video[:num_frames]
        else:
            return video[num_frames:]

    def upsample_downsample(self, video, prob=0.5):
        s = np.random.rand()
        if s > prob:
            return video
        s = np.random.randint(0, 1)
        if s:
            return self.upsample(video)
        else:
            return self.downsample(video)

    def temporal_elastic_transformation(self, video, prob=0.5):
        s = np.random.rand()
        if s > prob:
            return video
        return TemporalElasticTransformation()(video)

    def load_data(self, path):
        # filepath = os.path.join(os.getcwd(), 'guardians_of_children/violence_predictor/' + path)
        # data = np.load(filepath, mmap_mode='r')
        # load the processed .npy files
        data = np.load(path, mmap_mode='r')
        data = np.float32(data)
        # sampling frames uniformly from the entire video
        if self.sample:
            data = self.uniform_sampling(
                video=data, target_frames=self.target_frames)

        if self.mode == "both":
            frames = True
            differences = True
        elif self.mode == "only_frames":
            frames = True
            differences = False
        elif self.mode == "only_differences":
            frames = False
            differences = True

        # data augmentation
        if self.data_aug:
            data = self.random_brightness(data, (0.5, 1.5))
            data = self.color_jitter(data, prob=1)
            data = self.random_flip(data, prob=0.50)
            data = self.random_crop(data, prob=0.80)
            data = self.random_rotation(data, rg=25, prob=0.8)
            data = self.inverse_order(data, prob=0.15)
            data = self.upsample_downsample(data, prob=0.5)
            data = self.temporal_elastic_transformation(data, prob=0.2)
            data = self.gaussian_blur(data, prob=0.2, low=1, high=2)

            if differences:
                diff_data = self.frame_difference(data)
            if frames and self.background_suppress:
                data = self.background_suppression(data)  ####
                data = self.pepper(data, prob=0.3, ratio=45)
                data = self.salt(data, prob=0.3, ratio=45)
        else:
            if self.dataset == 'rwf2000' or self.dataset == 'surv':
                data = self.crop_center(data, x_crop=(320 - 224) // 2,
                                        y_crop=(320 - 224) // 2)  # center cropping only for test generators
            if differences:
                diff_data = self.frame_difference(data)
            if frames and self.background_suppress:
                data = self.background_suppression(data)  ####

        if frames:
            data = np.array(data, dtype=np.float32)
            if self.normalize_:
                data = self.normalize(data)
            assert (data.shape == (self.target_frames, self.resize, self.resize, 3)), str(data.shape)
        if differences:
            diff_data = np.array(diff_data, dtype=np.float32)
            if self.normalize_:
                diff_data = self.normalize(diff_data)
            assert (diff_data.shape == (
            self.target_frames - self.frame_diff_interval, self.resize, self.resize, 3)), str(data.shape)

        if self.mode == "both":
            return data, diff_data
        elif self.mode == "only_frames":
            return data
        elif self.mode == "only_differences":
            return diff_data


"""
###########################################################################################################
dataProcess.py
###########################################################################################################
"""
def crop_img_remove_black(img, x_crop, y_crop, y, x):
    x_start = x_crop
    x_end = x - x_crop
    y_start = y_crop
    y_end = y-y_crop
    frame = img[y_start:y_end, x_start:x_end, :]
    return frame

def uniform_sampling(video, target_frames=64):
    # get total frames of input video and calculate sampling interval
    len_frames = video.shape[0]
    interval = int(np.ceil(len_frames/target_frames))
    # init empty list for sampled video and
    sampled_video = []
    for i in range(0, len_frames, interval):
        sampled_video.append(video[i])
    # calculate numer of padded frames and fix it
    num_pad = target_frames - len(sampled_video)
    padding = []
    if num_pad > 0:
        for i in range(-num_pad, 0):
            try:
                padding.append(video[i])
            except:
                padding.append(video[0])
        sampled_video += padding
    # get sampled video
    return np.array(sampled_video)

def Video2Npy(file_path, resize=320, crop_x_y=None, target_frames=None):
    # Load video
    cap = cv2.VideoCapture(file_path)
    # Get number of frames
    len_frames = int(cap.get(7))
    frames = []
    try:
        for i in range(len_frames):
            _, x_ = cap.read()
            if crop_x_y:
                frame = crop_img_remove_black(
                    x_, crop_x_y[0], crop_x_y[1], x_.shape[0], x_.shape[1])
            else:
                frame = x_
            frame = cv2.resize(frame, (resize,resize), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (resize, resize, 3))
            frames.append(frame)
    except Exception as e:
        print("Error: ", file_path, len_frames)
        print(e)
    finally:
        frames = np.array(frames)
        cap.release()
    frames = uniform_sampling(frames, target_frames=target_frames)
    return frames

def Save2Npy(file_dir, crop_x_y=None, target_frames=None, frame_size=320):
    # file_path = os.path.join(os.getcwd(), glob.glob('guardians_of_children/violence_predictor/*.mp4')[0])
    file_path = os.path.join(os.getcwd(), glob.glob('*.mp4')[0])
    # Load and preprocess video
    data = Video2Npy(file_path=file_path, resize=frame_size,
                         crop_x_y=crop_x_y, target_frames=target_frames)
    if target_frames:
        assert (data.shape == (target_frames,
                                   frame_size, frame_size, 3))
    r_frame = data[np.random.randint(data.shape[0])]
    # np.save(os.path.join(os.getcwd(),"guardians_of_children/violence_predictor/processed.npy"), np.uint8(data))
    np.save(os.path.join(os.getcwd(),"processed.npy"), np.uint8(data))
    return r_frame

def convert_dataset_to_npy(src, crop_x_y=None, target_frames=None, frame_size=320):
    path1 = os.path.join(src)
    r_frame = Save2Npy(file_dir=path1, crop_x_y=crop_x_y, target_frames=target_frames, frame_size=frame_size)
    return r_frame


"""
###########################################################################################################
models.py
###########################################################################################################
"""


def getProposedModelC(size=224, seq_len=32, cnn_weight='imagenet', cnn_trainable=True, lstm_type='sepconv',
                      weight_decay=2e-5, frame_diff_interval=1, mode="both", cnn_dropout=0.25, lstm_dropout=0.25,
                      dense_dropout=0.3, seed=42):
    """parameters:
    size = height/width of each frame,
    seq_len = number of frames in each sequence,
    cnn_weight= None or 'imagenet'
    mode = "only_frames" or "only_differences" or "both"
       returns:
    model
    """
    if mode == "both":
        frames = True
        differences = True
    elif mode == "only_frames":
        frames = True
        differences = False
    elif mode == "only_differences":
        frames = False
        differences = True

    if frames:

        frames_input = Input(shape=(seq_len, size, size, 3), name='frames_input')
        frames_cnn = MobileNetV2(input_shape=(size, size, 3), alpha=0.35, weights='imagenet', include_top=False)
        frames_cnn = Model(inputs=[frames_cnn.layers[0].input],
                           outputs=[frames_cnn.layers[-30].output])  # taking only upto block 13

        for layer in frames_cnn.layers:
            layer.trainable = cnn_trainable

        frames_cnn = TimeDistributed(frames_cnn, name='frames_CNN')(frames_input)
        frames_cnn = TimeDistributed(LeakyReLU(alpha=0.1), name='leaky_relu_1_')(frames_cnn)
        frames_cnn = TimeDistributed(Dropout(cnn_dropout, seed=seed), name='dropout_1_')(frames_cnn)

        if lstm_type == 'sepconv':
            frames_lstm = SepConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False,
                                        dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='SepConvLSTM2D_1',
                                        kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(
                frames_cnn)
        elif lstm_type == 'conv':
            frames_lstm = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False,
                                     dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='ConvLSTM2D_1',
                                     kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(
                frames_cnn)
        elif lstm_type == 'asepconv':
            frames_lstm = AttenSepConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False,
                                             dropout=lstm_dropout, recurrent_dropout=lstm_dropout,
                                             name='AttenSepConvLSTM2D_1', kernel_regularizer=l2(weight_decay),
                                             recurrent_regularizer=l2(weight_decay))(frames_cnn)
        else:
            raise Exception("lstm type not recognized!")

        frames_lstm = BatchNormalization(axis=-1)(frames_lstm)

    if differences:

        frames_diff_input = Input(shape=(seq_len - frame_diff_interval, size, size, 3), name='frames_diff_input')
        frames_diff_cnn = MobileNetV2(input_shape=(size, size, 3), alpha=0.35, weights='imagenet', include_top=False)
        frames_diff_cnn = Model(inputs=[frames_diff_cnn.layers[0].input],
                                outputs=[frames_diff_cnn.layers[-30].output])  # taking only upto block 13

        for layer in frames_diff_cnn.layers:
            layer.trainable = cnn_trainable

        frames_diff_cnn = TimeDistributed(frames_diff_cnn, name='frames_diff_CNN')(frames_diff_input)
        frames_diff_cnn = TimeDistributed(LeakyReLU(alpha=0.1), name='leaky_relu_2_')(frames_diff_cnn)
        frames_diff_cnn = TimeDistributed(Dropout(cnn_dropout, seed=seed), name='dropout_2_')(frames_diff_cnn)

        if lstm_type == 'sepconv':
            frames_diff_lstm = SepConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False,
                                             dropout=lstm_dropout, recurrent_dropout=lstm_dropout,
                                             name='SepConvLSTM2D_2', kernel_regularizer=l2(weight_decay),
                                             recurrent_regularizer=l2(weight_decay))(frames_diff_cnn)
        elif lstm_type == 'conv':
            frames_diff_lstm = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False,
                                          dropout=lstm_dropout, recurrent_dropout=lstm_dropout, name='ConvLSTM2D_2',
                                          kernel_regularizer=l2(weight_decay), recurrent_regularizer=l2(weight_decay))(
                frames_diff_cnn)
        elif lstm_type == 'asepconv':
            frames_diff_lstm = AttenSepConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same',
                                                  return_sequences=False, dropout=lstm_dropout,
                                                  recurrent_dropout=lstm_dropout, name='AttenSepConvLSTM2D_2',
                                                  kernel_regularizer=l2(weight_decay),
                                                  recurrent_regularizer=l2(weight_decay))(frames_diff_cnn)
        else:
            raise Exception("lstm type not recognized!")

        frames_diff_lstm = BatchNormalization(axis=-1)(frames_diff_lstm)

    if frames:
        frames_lstm = MaxPooling2D((2, 2))(frames_lstm)
        x1 = Flatten()(frames_lstm)
        x1 = Dense(64)(x1)
        x1 = LeakyReLU(alpha=0.1)(x1)

    if differences:
        frames_diff_lstm = MaxPooling2D((2, 2))(frames_diff_lstm)
        x2 = Flatten()(frames_diff_lstm)
        x2 = Dense(64)(x2)
        x2 = LeakyReLU(alpha=0.1)(x2)

    if mode == "both":
        x = Concatenate(axis=-1)([x1, x2])
    elif mode == "only_frames":
        x = x1
    elif mode == "only_differences":
        x = x2

    x = Dropout(dense_dropout, seed=seed)(x)
    x = Dense(16)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dense_dropout, seed=seed)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    if mode == "both":
        model = Model(inputs=[frames_input, frames_diff_input], outputs=predictions)
    elif mode == "only_frames":
        model = Model(inputs=frames_input, outputs=predictions)
    elif mode == "only_differences":
        model = Model(inputs=frames_diff_input, outputs=predictions)

    return model

"""
###########################################################################################################
sep_conv_rnn.py
###########################################################################################################
"""
class SepConvRNN2D(RNN):
    def __init__(self,
                 cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if unroll:
            raise TypeError('Unrolling isn\'t possible with '
                            'convolutional RNNs.')
        if isinstance(cell, (list, tuple)):
            # The StackedConvRNN2DCells isn't implemented yet.
            raise TypeError('It is not possible at the moment to'
                            'stack convolutional cells.')
        super(SepConvRNN2D, self).__init__(cell,
                                           return_sequences,
                                           return_state,
                                           go_backwards,
                                           stateful,
                                           unroll,
                                           **kwargs)
        self.input_spec = [InputSpec(ndim=5)]
        self.states = None
        self._num_constants = None

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        cell = self.cell
        if cell.data_format == 'channels_first':
            rows = input_shape[3]
            cols = input_shape[4]
        elif cell.data_format == 'channels_last':
            rows = input_shape[2]
            cols = input_shape[3]
        rows = conv_utils.conv_output_length(rows,
                                             cell.kernel_size[0],
                                             padding=cell.padding,
                                             stride=cell.strides[0],
                                             dilation=cell.dilation_rate[0])
        cols = conv_utils.conv_output_length(cols,
                                             cell.kernel_size[1],
                                             padding=cell.padding,
                                             stride=cell.strides[1],
                                             dilation=cell.dilation_rate[1])

        if cell.data_format == 'channels_first':
            output_shape = input_shape[:2] + (cell.filters, rows, cols)
        elif cell.data_format == 'channels_last':
            output_shape = input_shape[:2] + (rows, cols, cell.filters)

        if not self.return_sequences:
            output_shape = output_shape[:1] + output_shape[2:]

        if self.return_state:
            output_shape = [output_shape]
            if cell.data_format == 'channels_first':
                output_shape += [(input_shape[0], cell.filters, rows, cols)
                                 for _ in range(2)]
            elif cell.data_format == 'channels_last':
                output_shape += [(input_shape[0], rows, cols, cell.filters)
                                 for _ in range(2)]
        return output_shape

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states and
        # constants if these are passed in __call__.
        if self._num_constants is not None:
            constants_shape = input_shape[-self._num_constants:]  # pylint: disable=E1130
        else:
            constants_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:5])

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            if constants_shape is not None:
                self.cell.build([step_input_shape] + constants_shape)
            else:
                self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if self.cell.data_format == 'channels_first':
                ch_dim = 1
            elif self.cell.data_format == 'channels_last':
                ch_dim = 3
            if [spec.shape[ch_dim] for spec in self.state_spec] != state_size:
                raise ValueError(
                    'An initial_state was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'However `cell.state_size` is '
                    '{}'.format([spec.shape for spec in self.state_spec],
                                self.cell.state_size))
        else:
            if self.cell.data_format == 'channels_first':
                self.state_spec = [InputSpec(shape=(None, dim, None, None))
                                   for dim in state_size]
            elif self.cell.data_format == 'channels_last':
                self.state_spec = [InputSpec(shape=(None, None, None, dim))
                                   for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_state(self, inputs):
        # (samples, timesteps, rows, cols, filters)
        initial_state = K.zeros_like(inputs)
        # (samples, rows, cols, filters)
        initial_state = K.sum(initial_state, axis=1)
        depth_shape = list(self.cell.depth_kernel_shape)
        depth_shape[-1] = self.cell.depth_multiplier
        point_shape = list(self.cell.point_kernel_shape)
        point_shape[-1] = self.cell.filters
        initial_state = self.cell.input_conv(initial_state,
                                             array_ops.zeros(tuple(depth_shape)), array_ops.zeros(tuple(point_shape)),
                                             padding=self.cell.padding)

        if hasattr(self.cell.state_size, '__len__'):
            return [initial_state for _ in self.cell.state_size]
        else:
            return [initial_state]

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = _standardize_args(
            inputs, initial_state, constants, self._num_constants)

        if initial_state is None and constants is None:
            return super(SepConvRNN2D, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = []
            for state in initial_state:
                shape = K.int_shape(state)
                self.state_spec.append(InputSpec(shape=shape))

            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != K.is_keras_tensor(additional_inputs[0]):
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors')

        if K.is_keras_tensor(additional_inputs[0]):
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(SepConvRNN2D, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return super(SepConvRNN2D, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None, constants=None):
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            inputs = inputs[0]
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        timesteps = K.int_shape(inputs)[1]

        kwargs = {}
        if generic_utils.has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not generic_utils.has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = K.rnn(step,
                                             inputs,
                                             initial_state,
                                             constants=constants,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append(K.update(self.states[i], states[i]))
            self.add_update(updates)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        if self.return_state:
            if not isinstance(states, (list, tuple)):
                states = [states]
            else:
                states = list(states)
            return [output] + states
        else:
            return output

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        input_shape = self.input_spec[0].shape
        state_shape = self.compute_output_shape(input_shape)
        if self.return_state:
            state_shape = state_shape[0]
        if self.return_sequences:
            state_shape = state_shape[:1].concatenate(state_shape[2:])
        if None in state_shape:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.\n'
                             'The same thing goes for the number of rows and '
                             'columns.')

        # helper function
        def get_tuple_shape(nb_channels):
            result = list(state_shape)
            if self.cell.data_format == 'channels_first':
                result[1] = nb_channels
            elif self.cell.data_format == 'channels_last':
                result[3] = nb_channels
            else:
                raise KeyError
            return tuple(result)

        # initialize state if None
        if self.states[0] is None:
            if hasattr(self.cell.state_size, '__len__'):
                self.states = [K.zeros(get_tuple_shape(dim))
                               for dim in self.cell.state_size]
            else:
                self.states = [K.zeros(get_tuple_shape(self.cell.state_size))]
        elif states is None:
            if hasattr(self.cell.state_size, '__len__'):
                for state, dim in zip(self.states, self.cell.state_size):
                    K.set_value(state, np.zeros(get_tuple_shape(dim)))
            else:
                K.set_value(self.states[0],
                            np.zeros(get_tuple_shape(self.cell.state_size)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, ' +
                                 'but it received ' + str(len(states)) +
                                 ' state values. Input received: ' + str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, '__len__'):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != get_tuple_shape(dim):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str(get_tuple_shape(dim)) +
                                     ', found shape=' + str(value.shape))
                # TODO(anjalisridhar): consider batch calls to `set_value`.
                K.set_value(state, value)


class SepConvLSTM2DCell(DropoutRNNCellMixin, Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 depth_multiplier=1,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(SepConvLSTM2DCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.depth_multiplier = depth_multiplier
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters, self.filters)

    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        depth_kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier * 4)
        point_kernel_shape = (1, 1) + (input_dim * self.depth_multiplier, self.filters * 4)

        self.depth_kernel_shape = depth_kernel_shape
        self.point_kernel_shape = point_kernel_shape

        recurrent_depth_kernel_shape = self.kernel_size + (self.filters, self.depth_multiplier * 4)
        recurrent_point_kernel_shape = (1, 1) + (self.filters * self.depth_multiplier, self.filters * 4)

        self.depth_kernel_shape = depth_kernel_shape
        self.point_kernel_shape = point_kernel_shape

        self.depth_kernel = self.add_weight(shape=depth_kernel_shape,
                                            initializer=self.kernel_initializer,
                                            name='depth_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)

        self.point_kernel = self.add_weight(shape=point_kernel_shape,
                                            initializer=self.kernel_initializer,
                                            name='point_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)

        self.recurrent_depth_kernel = self.add_weight(
            shape=recurrent_depth_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_depth_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.recurrent_point_kernel = self.add_weight(
            shape=recurrent_point_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_point_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.filters,), *args, **kwargs),
                        initializers.Ones()((self.filters,), *args, **kwargs),
                        self.bias_initializer((self.filters * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.filters * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        # dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        (depth_kernel_i, depth_kernel_f,
         depth_kernel_c, depth_kernel_o) = array_ops.split(self.depth_kernel, 4, axis=3)
        (recurrent_depth_kernel_i,
         recurrent_depth_kernel_f,
         recurrent_depth_kernel_c,
         recurrent_depth_kernel_o) = array_ops.split(self.recurrent_depth_kernel, 4, axis=3)

        (point_kernel_i, point_kernel_f,
         point_kernel_c, point_kernel_o) = array_ops.split(self.point_kernel, 4, axis=3)
        (recurrent_point_kernel_i,
         recurrent_point_kernel_f,
         recurrent_point_kernel_c,
         recurrent_point_kernel_o) = array_ops.split(self.recurrent_point_kernel, 4, axis=3)

        if self.use_bias:
            bias_i, bias_f, bias_c, bias_o = array_ops.split(self.bias, 4)
        else:
            bias_i, bias_f, bias_c, bias_o = None, None, None, None

        x_i = self.input_conv(inputs_i, depth_kernel_i, point_kernel_i, bias_i, padding=self.padding)
        x_f = self.input_conv(inputs_f, depth_kernel_f, point_kernel_f, bias_f, padding=self.padding)
        x_c = self.input_conv(inputs_c, depth_kernel_c, point_kernel_c, bias_c, padding=self.padding)
        x_o = self.input_conv(inputs_o, depth_kernel_o, point_kernel_o, bias_o, padding=self.padding)
        h_i = self.recurrent_conv(h_tm1_i, recurrent_depth_kernel_i, recurrent_point_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, recurrent_depth_kernel_f, recurrent_point_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, recurrent_depth_kernel_c, recurrent_point_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, recurrent_depth_kernel_o, recurrent_point_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)
        return h, [h, c]

    def input_conv(self, x, dw, pw, b=None, padding='valid'):
        conv_out = K.separable_conv2d(x, dw, pw, strides=self.strides,
                                      padding=padding,
                                      data_format=self.data_format,
                                      dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, dw, pw):
        conv_out = K.separable_conv2d(x, dw, pw, strides=(1, 1),
                                      padding='same',
                                      data_format=self.data_format)
        return conv_out

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'depth_multiplier': self.depth_multiplier,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SepConvLSTM2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.SepConvLSTM2D')
class SepConvLSTM2D(SepConvRNN2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 depth_multiplier=1,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        cell = SepConvLSTM2DCell(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 data_format=data_format,
                                 dilation_rate=dilation_rate,
                                 depth_multiplier=depth_multiplier,
                                 activation=activation,
                                 recurrent_activation=recurrent_activation,
                                 use_bias=use_bias,
                                 kernel_initializer=kernel_initializer,
                                 recurrent_initializer=recurrent_initializer,
                                 bias_initializer=bias_initializer,
                                 unit_forget_bias=unit_forget_bias,
                                 kernel_regularizer=kernel_regularizer,
                                 recurrent_regularizer=recurrent_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 kernel_constraint=kernel_constraint,
                                 recurrent_constraint=recurrent_constraint,
                                 bias_constraint=bias_constraint,
                                 dropout=dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 dtype=kwargs.get('dtype'))
        super(SepConvLSTM2D, self).__init__(cell,
                                            return_sequences=return_sequences,
                                            go_backwards=go_backwards,
                                            stateful=stateful,
                                            **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(SepConvLSTM2D, self).call(inputs,
                                               mask=mask,
                                               training=training,
                                               initial_state=initial_state)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def depth_multiplier(self):
        return self.cell.depth_multiplier

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'depth_multiplier': self.depth_multiplier,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SepConvLSTM2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AttenSepConvLSTM2DCell(DropoutRNNCellMixin, Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 depth_multiplier=1,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(AttenSepConvLSTM2DCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.depth_multiplier = depth_multiplier
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters, self.filters)

    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        # print('>>>>>>>>>>>>>>>> ', input_shape)
        self.feat_shape = input_shape  # (input_shape[0], input_shape[1], input_shape[2], input_shape[3])
        input_dim = input_shape[channel_axis]
        depth_kernel_shape = self.kernel_size + (input_dim, self.depth_multiplier * 4)
        point_kernel_shape = (1, 1) + (input_dim * self.depth_multiplier, self.filters * 4)
        depth_kernel_a_shape = self.kernel_size + (input_dim, self.depth_multiplier)
        point_kernel_a_shape = (1, 1) + (input_dim * self.depth_multiplier, input_dim)

        self.depth_kernel_shape = depth_kernel_shape
        self.point_kernel_shape = point_kernel_shape

        recurrent_depth_kernel_shape = self.kernel_size + (self.filters, self.depth_multiplier * 4)
        recurrent_point_kernel_shape = (1, 1) + (self.filters * self.depth_multiplier, self.filters * 4)
        recurrent_depth_kernel_a_shape = self.kernel_size + (self.filters, self.depth_multiplier)
        recurrent_point_kernel_a_shape = (1, 1) + (self.filters * self.depth_multiplier, input_dim)

        self.recurrent_depth_kernel_shape = depth_kernel_shape
        self.recurrent_point_kernel_shape = point_kernel_shape

        self.depth_kernel = self.add_weight(shape=depth_kernel_shape,
                                            initializer=self.kernel_initializer,
                                            name='depth_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)

        self.point_kernel = self.add_weight(shape=point_kernel_shape,
                                            initializer=self.kernel_initializer,
                                            name='point_kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)

        self.depth_kernel_a = self.add_weight(shape=depth_kernel_a_shape,
                                              initializer=self.kernel_initializer,
                                              name='depth_kernel_a',
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        self.point_kernel_a = self.add_weight(shape=point_kernel_a_shape,
                                              initializer=self.kernel_initializer,
                                              name='point_kernel_a',
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)

        self.recurrent_depth_kernel = self.add_weight(
            shape=recurrent_depth_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_depth_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.recurrent_point_kernel = self.add_weight(
            shape=recurrent_point_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_point_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        self.recurrent_depth_kernel_a = self.add_weight(
            shape=recurrent_depth_kernel_a_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_depth_kernel_a',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.recurrent_point_kernel_a = self.add_weight(
            shape=recurrent_point_kernel_a_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_point_kernel_a',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.attention_weight = self.add_weight(
            shape=self.kernel_size + (input_dim, 1),
            initializer=self.kernel_initializer,
            name='attention_weight',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.filters,), *args, **kwargs),
                        initializers.Ones()((self.filters,), *args, **kwargs),
                        self.bias_initializer((self.filters * 2,), *args, **kwargs),
                    ])

            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.filters * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
            self.bias_a = self.add_weight(
                shape=(input_dim,),
                name='bias_a',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        # dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        (depth_kernel_i, depth_kernel_f,
         depth_kernel_c, depth_kernel_o) = array_ops.split(self.depth_kernel, 4, axis=3)
        (recurrent_depth_kernel_i,
         recurrent_depth_kernel_f,
         recurrent_depth_kernel_c,
         recurrent_depth_kernel_o) = array_ops.split(self.recurrent_depth_kernel, 4, axis=3)

        (point_kernel_i, point_kernel_f,
         point_kernel_c, point_kernel_o) = array_ops.split(self.point_kernel, 4, axis=3)
        (recurrent_point_kernel_i,
         recurrent_point_kernel_f,
         recurrent_point_kernel_c,
         recurrent_point_kernel_o) = array_ops.split(self.recurrent_point_kernel, 4, axis=3)

        if self.use_bias:
            bias_i, bias_f, bias_c, bias_o = array_ops.split(self.bias, 4)
        else:
            bias_i, bias_f, bias_c, bias_o = None, None, None, None

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
        else:
            inputs_i = inputs
        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
        else:
            h_tm1_i = h_tm1

        x_a = self.input_conv(inputs_i, self.depth_kernel_a, self.point_kernel_a, self.bias_a, padding=self.padding)
        h_a = self.recurrent_conv(h_tm1_i, self.recurrent_depth_kernel_a, self.recurrent_point_kernel_a)
        inputs = inputs * self.attention(x_a + h_a, self.attention_weight)
        if 0 < self.dropout < 1.:
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
        if 0 < self.recurrent_dropout < 1.:
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        x_i = self.input_conv(inputs, depth_kernel_i, point_kernel_i, bias_i, padding=self.padding)
        x_f = self.input_conv(inputs_f, depth_kernel_f, point_kernel_f, bias_f, padding=self.padding)
        x_c = self.input_conv(inputs_c, depth_kernel_c, point_kernel_c, bias_c, padding=self.padding)
        x_o = self.input_conv(inputs_o, depth_kernel_o, point_kernel_o, bias_o, padding=self.padding)
        h_i = self.recurrent_conv(h_tm1, recurrent_depth_kernel_i, recurrent_point_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, recurrent_depth_kernel_f, recurrent_point_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, recurrent_depth_kernel_c, recurrent_point_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, recurrent_depth_kernel_o, recurrent_point_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)
        return h, [h, c]

    def input_conv(self, x, dw, pw, b=None, padding='valid'):
        conv_out = K.separable_conv2d(x, dw, pw, strides=self.strides,
                                      padding=padding,
                                      data_format=self.data_format,
                                      dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, dw, pw):
        conv_out = K.separable_conv2d(x, dw, pw, strides=(1, 1),
                                      padding='same',
                                      data_format=self.data_format)
        return conv_out

    def attention(self, x, w):
        z = K.conv2d(K.tanh(x),
                     w,
                     strides=self.strides,
                     padding=self.padding,
                     data_format=self.data_format,
                     dilation_rate=self.dilation_rate)
        shape_2d = tf.convert_to_tensor([-1, self.feat_shape[1], self.feat_shape[2], 1])
        shape_1d = tf.convert_to_tensor([-1, self.feat_shape[1] * self.feat_shape[2]])
        att = K.softmax(K.reshape(z, shape_1d))
        att = K.reshape(att, shape_2d)
        return K.repeat_elements(att, self.feat_shape[3], 3)

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'depth_multiplier': self.depth_multiplier,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(AttenSepConvLSTM2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.AttenSepConvLSTM2D')
class AttenSepConvLSTM2D(SepConvRNN2D):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 depth_multiplier=1,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        cell = AttenSepConvLSTM2DCell(filters=filters,
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      data_format=data_format,
                                      dilation_rate=dilation_rate,
                                      depth_multiplier=depth_multiplier,
                                      activation=activation,
                                      recurrent_activation=recurrent_activation,
                                      use_bias=use_bias,
                                      kernel_initializer=kernel_initializer,
                                      recurrent_initializer=recurrent_initializer,
                                      bias_initializer=bias_initializer,
                                      unit_forget_bias=unit_forget_bias,
                                      kernel_regularizer=kernel_regularizer,
                                      recurrent_regularizer=recurrent_regularizer,
                                      bias_regularizer=bias_regularizer,
                                      kernel_constraint=kernel_constraint,
                                      recurrent_constraint=recurrent_constraint,
                                      bias_constraint=bias_constraint,
                                      dropout=dropout,
                                      recurrent_dropout=recurrent_dropout,
                                      dtype=kwargs.get('dtype'))
        super(AttenSepConvLSTM2D, self).__init__(cell,
                                                 return_sequences=return_sequences,
                                                 go_backwards=go_backwards,
                                                 stateful=stateful,
                                                 **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(AttenSepConvLSTM2D, self).call(inputs,
                                                    mask=mask,
                                                    training=training,
                                                    initial_state=initial_state)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def depth_multiplier(self):
        return self.cell.depth_multiplier

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'depth_multiplier': self.depth_multiplier,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(AttenSepConvLSTM2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


"""
###########################################################################################################
qulitative.py
###########################################################################################################
"""

def qualitative():
    mode = "both"
    dataset = 'custom'
    vid_len = 32
    dataset_frame_size = 224
    input_frame_size = 224
    frame_diff_interval = 1
    one_hot = False
    lstm_type = 'sepconv' 

    r_frame = convert_dataset_to_npy(src='./'.format(dataset), crop_x_y=None, target_frames=vid_len,
                                     frame_size= dataset_frame_size)

    test_generator = DataGenerator(X_path='processed.npy'.format(dataset),
                                batch_size=1,
                                data_augmentation=False,
                                shuffle=True,
                                one_hot=one_hot,
                                sample=False,
                                resize=input_frame_size,
                                target_frames = vid_len,
                                frame_diff_interval = frame_diff_interval,
                                dataset = dataset,
                                normalize_ = False,
                                background_suppress = False,
                                mode = mode)

    model =  getProposedModelC(size=224, seq_len=32, frame_diff_interval = 1, mode="both", lstm_type=lstm_type)
    model.load_weights('./ckpt/rwf2000_best_val_acc_Model').expect_partial()
    # model.load_weights(os.path.join(os.getcwd(),'guardians_of_children/violence_predictor/ckpt_all/rwf2000_currentModel')).expect_partial()
    model.trainable = False
    abuse_perc = evaluate(model, test_generator)
    return abuse_perc, r_frame


def evaluate(model, datagen):
    for i, (x,y) in enumerate(datagen):
        p = model.predict(x)
        p = np.squeeze(p)
        abuse_perc = 1 - p
        return abuse_perc

def calculate():
    abuse_perc, r_frame = qualitative()
    print(abuse_perc)
    if abuse_perc >= 0.75 :
        print(abuse_perc, r_frame.shape)
        # cv2.imwrite("random_frame.jpg", r_frame)
        return abuse_perc, r_frame
    return abuse_perc

calculate()