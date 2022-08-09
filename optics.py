import abc

import tensorflow as tf
import numpy as np

from numpy.fft import fftshift, ifftshift
import fractions

import cv2
import os


def tf_compl_exp(phase, dtype=tf.complex64, name='complex_exp'):
    """
    Adapted from [Sitzmann et al. 2018]
    phase is NOT normalized and should range from -pi to pi
    """
    if dtype==tf.complex64:
        phase = tf.cast(phase, tf.float32)
    else: # tf.complex128
        phase = tf.cast(phase, tf.float64)
    return tf.complex(tf.cos(phase), tf.sin(phase), name=name)


def tf_compl_val(amplitude, phase, dtype=tf.complex64, name='complex_val'):
    """
    Adapted from [Sitzmann et al. 2018]
    phase is NOT normalized and should range from -pi to pi
    """
    if dtype==tf.complex64:
        amplitude = tf.cast(amplitude, tf.float32)
        phase = tf.cast(phase, tf.float32)
    else: # tf.complex128
        amplitude = tf.cast(amplitude, tf.float64)
        phase = tf.cast(phase, tf.float64)
    return tf.complex(amplitude*tf.cos(phase), amplitude*tf.sin(phase), name=name)


def tf_fft2d(a_tensor, dtype=tf.complex64):
    """
    Adapted from [Sitzmann et al. 2018]
    takes images of shape [batch_size, channels, x, y] and apply fft2
    """
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    if not a_tensor.dtype.is_complex:
        a_tensor = tf.complex(a_tensor, 0.)
    a_fft2d = tf.signal.fft2d(a_tensor)
    if not a_fft2d.dtype == dtype:
        a_fft2d = tf.cast(a_fft2d, dtype)
    return a_fft2d


def tf_ifft2d(a_tensor, dtype=tf.complex64):
    """
    Adapted from [Sitzmann et al. 2018]
    takes images of shape [batch_size, channels, x, y] and apply ifft2
    """
    if not a_tensor.dtype.is_complex:
        a_tensor = tf.complex(a_tensor, 0.)
    a_ifft2d = tf.signal.ifft2d(a_tensor)
    if not a_ifft2d.dtype == dtype:
        a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d


def tf_fftshift2d(a_tensor, input_shape=None):
    """
    Adapted from [Sitzmann et al. 2018]
    a_tensor has to be NCHW
    """
    if not input_shape:
        input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(2, 4):
        split = (input_shape[axis] + 1) // 2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def tf_ifftshift2d(a_tensor, input_shape=None):
    """
    Adapted from [Sitzmann et al. 2018]
    a_tensor has to be NCHW
    """
    if not input_shape:
        input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(2, 4):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor    

def tf_wrap_phs(phs_only, 
                phs_max=[2*np.pi]*3, 
                adaptive_phs_shift=False):
    def wrap_less_equal_than_phs_max(phs_only, phs_max, phs_per_channel_max, phs_per_channel_min):
        return phs_only + (phs_max-phs_per_channel_min-phs_per_channel_max) / 2.0
    
    def wrap_greater_than_phs_max(phs_only, phs_max):
        phs_only = phs_only + phs_max/2.0
        phs_only = tf.where(phs_only < 0, phs_only + 2.0*np.pi, phs_only)
        phs_only = tf.where(phs_only > phs_max, phs_only - 2.0*np.pi, phs_only)
        return phs_only

    if phs_max is not None:
        # wrap out-of-range phase
        if adaptive_phs_shift:
            phs_per_channel_list = []
            for i in range(3):
                phase_per_channel = phs_only[:,i,:,:]
                phs_max_channel = phs_max[i]
                phase_per_channel_max = tf.reduce_max(phase_per_channel)
                phase_per_channel_min = tf.reduce_min(phase_per_channel)
                phase_per_channel = tf.cond((phase_per_channel_max - phase_per_channel_min) <= phs_max_channel, 
                    lambda: wrap_less_equal_than_phs_max(phase_per_channel, phs_max_channel, phase_per_channel_max, phase_per_channel_min), 
                    lambda: wrap_greater_than_phs_max(phase_per_channel, phs_max_channel))
                phs_per_channel_list.append(phase_per_channel)
            phs_only = tf.stack(phs_per_channel_list, axis=1)
        else:
            phs_max_4d = np.reshape(phs_max, [1,3,1,1])
            phs_only = wrap_greater_than_phs_max(phs_only, phs_max_4d) 
    
    return phs_only

# add propagation, check normalize
def tf_dpm_maimone(cpx, 
                   propagator=None,
                   depth_shift=0,
                   adaptive_phs_shift=False,
                   batch=1, 
                   num_channels=3, 
                   res_h=384, 
                   res_w=384,
                   axis=2,
                   phs_max=[2*np.pi]*3, 
                   amp_max=None, 
                   clamp=False,
                   normalize=True,
                   wavelength=[0.000450, 0.000520, 0.000638]):
    """
    Double phase method of [Maimone et al. 2017]
    """
    # shift the hologram to hologram plane
    assert (depth_shift == 0 or propagator != None)
    if depth_shift != 0:
        tf_wavelength = tf.constant(np.array(wavelength).reshape(1,3,1,1))
        cpx = propagator(cpx, depth_shift) * tf_compl_exp(-2*np.pi*depth_shift/tf_wavelength)

    amp = tf.abs(cpx)
    phs = tf.math.angle(cpx)

    # normalize amplitude
    if amp_max is None:
        # avoid acos producing nan
        amp_max = tf.reduce_max(amp) + 1e-6
    amp = amp / amp_max

    # clamp maximum value to 1.0
    if clamp:
        amp = tf.minimum(amp, 1.0-1e-6)

    # center phase for each color channel
    phs_zero_mean = phs - tf.reduce_mean(phs, [2,3], keepdims=True)

    # discard every other pixel
    if axis == 3:    # reduce columns
        amp = amp[:,:,:,0::2]
        phs_zero_mean = phs_zero_mean[:,:,:,0::2]
    elif axis == 2:  # reduce rows
        amp = amp[:,:,0::2,:]
        phs_zero_mean = phs_zero_mean[:,:,0::2,:]

    # compute two phase maps
    phs_offset = tf.acos(amp)
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset 

    # arrange in checkerboard pattern
    if axis == 3:
        phs_1_1 = phs_low[:,:,0::2,:]
        phs_1_2 = phs_high[:,:,0::2,:]
        phs_2_1 = phs_high[:,:,1::2,:]
        phs_2_2 = phs_low[:,:,1::2,:]
        phs_only = tf.concat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], axis=1)
    elif axis == 2:
        phs_1_1 = phs_low[:,:,:,0::2]
        phs_1_2 = phs_high[:,:,:,0::2]
        phs_2_1 = phs_high[:,:,:,1::2]
        phs_2_2 = phs_low[:,:,:,1::2]
        phs_only = tf.concat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], axis=1)
    else:
        raise ValueError("axis has to be 2 or 3")
    phs_only = tf.depth_to_space(phs_only, 2, data_format='NCHW')

    if phs_max != None:
        phs_only = tf_wrap_phs(phs_only, phs_max=phs_max, adaptive_phs_shift=adaptive_phs_shift)

    if normalize:
        phs_max_4d = np.reshape(phs_max, [1,3,1,1])
        phs_only = phs_only / phs_max_4d

    return phs_only, amp_max


def np_gaussian(shape=(3,3), sigma=0.5, reshape_4d=True):
    """
    modified from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    2D Gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    
    if reshape_4d:
        h = h[:,:,np.newaxis,np.newaxis]
        h = h.repeat(3,2)

    return h


def tf_bldpm(cpx, 
             propagator=None,
             depth_shift=0,
             adaptive_phs_shift=False,
             batch=1, 
             num_channels=3, 
             res_h=384, 
             res_w=384,
             k=0.5,
             phs_max=[2*np.pi]*3, 
             amp_max=None, 
             clamp=False,
             normalize=True,
             wavelength=[0.000450, 0.000520, 0.000638]):
    """
    Band-limited double phase method [Sui et al. 2021]
    """
    # make grid
    square_filter = True
    y = tf.range(-(res_h // 2), res_h // 2, delta=1)
    x = tf.range(-(res_w // 2), res_w // 2, delta=1)
    x, y = tf.meshgrid(x, y)
    x = tf.stack([x]*3, axis=0)
    y = tf.stack([y]*3, axis=0)
    if square_filter:
        side_min = np.minimum(res_w, res_h)
        x = x[tf.newaxis,:,:,:] / side_min
        y = y[tf.newaxis,:,:,:] / side_min

    else:
        x = x[tf.newaxis,:,:,:] / res_w
        y = y[tf.newaxis,:,:,:] / res_h
    
    # Spatial frequency
    tan_pi_alpha_u = tf.tan(y * np.pi)
    tan_pi_alpha_mu = tf.tan(x * np.pi)
    mask = (tf.abs(tan_pi_alpha_u*tan_pi_alpha_mu) <= k)
    if square_filter:
        mask_two = tf.logical_and(tf.abs(x) <= 0.5, tf.abs(y) <= 0.5)
        mask = tf.logical_and(mask, mask_two)
    mask = tf.cast(mask, cpx.dtype)

    # shift the hologram to hologram plane
    assert (depth_shift == 0 or propagator != None)
    if depth_shift != 0:
        tf_wavelength = tf.constant(np.array(wavelength).reshape(1,3,1,1))
        cpx = propagator(cpx, depth_shift) * tf_compl_exp(-2*np.pi*depth_shift/tf_wavelength)

    # filter the complex hologram in frequency space
    cpx_fft = tf_fftshift2d(tf_fft2d(cpx)) * mask
    cpx = tf_ifft2d(tf_ifftshift2d(cpx_fft))
    amp = tf.abs(cpx)
    phs = tf.math.angle(cpx)
    
    # normalize amplitude
    if amp_max is None:
        # avoid acos producing nan
        amp_max = tf.reduce_max(amp) + 1e-6
    amp = amp / amp_max

    # clamp maximum value to 1.0
    if clamp:
        amp = tf.minimum(amp, 1.0-1e-6)

    # center phase for each color channel
    phs_zero_mean = phs - tf.reduce_mean(phs, [2,3], keepdims=True)

    # compute two phase maps
    phs_offset = tf.acos(amp)
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset

    # arrange in checkerboard pattern
    phs_1_1 = phs_low[:,:,0::2,0::2]
    phs_1_2 = phs_high[:,:,0::2,1::2]
    phs_2_1 = phs_high[:,:,1::2,0::2]
    phs_2_2 = phs_low[:,:,1::2,1::2]
    phs_only = tf.concat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], axis=1)
    phs_only = tf.compat.v1.depth_to_space(phs_only, 2, data_format='NCHW')
    
    if phs_max != None:
        phs_only = tf_wrap_phs(phs_only, phs_max=phs_max, adaptive_phs_shift=adaptive_phs_shift)

    if normalize:
        phs_max_4d = np.reshape(phs_max, [1,3,1,1])
        phs_only = phs_only / phs_max_4d

    # apply your own lookup table if necessary

    return phs_only, amp_max


def tf_aadpm(cpx, 
             propagator=None,
             depth_shift=0,
             adaptive_phs_shift=False,
             batch=1, 
             num_channels=3, 
             res_h=384, 
             res_w=384,
             sigma=0.5, 
             kernel_width=5, 
             phs_max=[2*np.pi]*3, 
             amp_max=None, 
             clamp=False,
             normalize=True,
             wavelength=[0.000450, 0.000520, 0.000638]):
    """
    Anti-aliasing double phase method
    """
    # shift the hologram to hologram plane
    assert (depth_shift == 0 or propagator != None)
    if depth_shift != 0:
        tf_wavelength = tf.constant(np.array(wavelength).reshape(1,3,1,1))
        cpx = propagator(cpx, depth_shift) * tf_compl_exp(-2*np.pi*depth_shift/tf_wavelength)

    # apply pre-blur
    if sigma > 0.0:
        blur_kernel = tf.convert_to_tensor(np_gaussian([kernel_width, kernel_width], sigma), dtype=tf.float32)
        cpx_imag = tf.math.imag(cpx)
        cpx_real = tf.math.real(cpx)
        cpx_imag = tf.nn.depthwise_conv2d(cpx_imag, blur_kernel, strides=[1,1,1,1], padding='SAME', data_format='NCHW')
        cpx_real = tf.nn.depthwise_conv2d(cpx_real, blur_kernel, strides=[1,1,1,1], padding='SAME', data_format='NCHW')
        cpx = tf.complex(cpx_real, cpx_imag)
    amp = tf.abs(cpx)
    phs = tf.math.angle(cpx)

    # normalize amplitude
    if amp_max is None:
        # avoid acos producing nan
        amp_max = tf.reduce_max(amp) + 1e-6
    amp = amp / amp_max

    # clamp maximum value to 1.0
    if clamp:
        amp = tf.minimum(amp, 1.0-1e-6)

    # center phase for each color channel
    phs_zero_mean = phs - tf.reduce_mean(phs, [2,3], keepdims=True)

    # compute two phase maps
    phs_offset = tf.acos(amp)
    phs_low = phs_zero_mean - phs_offset
    phs_high = phs_zero_mean + phs_offset 

    # arrange in checkerboard pattern
    phs_1_1 = phs_low[:,:,0::2,0::2]
    phs_1_2 = phs_high[:,:,0::2,1::2]
    phs_2_1 = phs_high[:,:,1::2,0::2]
    phs_2_2 = phs_low[:,:,1::2,1::2]
    phs_only = tf.concat([phs_1_1, phs_1_2, phs_2_1, phs_2_2], axis=1)
    phs_only = tf.compat.v1.depth_to_space(phs_only, 2, data_format='NCHW')
    
    if phs_max != None:
        phs_only = tf_wrap_phs(phs_only, phs_max=phs_max, adaptive_phs_shift=adaptive_phs_shift)

    if normalize:
        phs_max_4d = np.reshape(phs_max, [1,3,1,1])
        phs_only = phs_only / phs_max_4d

    # apply your own lookup table if necessary

    return phs_only, amp_max


def np_circ_filter(batch,
                   num_channels,
                   res_h,
                   res_w,
                   filter_radius,
                   ):
    """create a circular low pass filter
    """
    y,x = np.meshgrid(np.linspace(-(res_w-1)/2, (res_w-1)/2, res_w), np.linspace(-(res_h-1)/2, (res_h-1)/2, res_h))
    mask = x**2+y**2 <= filter_radius**2
    np_filter = np.zeros((res_h, res_w))
    np_filter[mask] = 1.0
    np_filter = np.tile(np.reshape(np_filter, [1,1,res_h,res_w]), [batch, num_channels, 1, 1])
    return np_filter


def tf_filter_phs_only(phs_only,
                       unnormalize_input=False,
                       normalize_output=True,
                       propagator=None, 
                       depth_shift=0, 
                       batch=2, 
                       num_channels=3, 
                       res_h=384, 
                       res_w=384, 
                       radius=None,
                       phs_max=[2*np.pi]*3, 
                       amp_max=1.0, 
                       wavelength=[0.000450, 0.000520, 0.000638]):
    """filter double-phase encoded phase-only hologram by modeling the physical aperture
    """
    # train mode
    if radius == None:
        radius = np.minimum(res_h, res_w) // 2

    # turn phs_only to complex holograms
    if unnormalize_input:
        phs_max = np.array(phs_max).reshape(1,3,1,1)
        phs_only = (phs_only-0.5) * phs_max
    phs_only_cpx = tf_compl_val(tf.ones_like(phs_only), phs_only) * tf.cast(amp_max, tf.complex64)

    # converts to fourier space and low-pass filter
    phs_only_cpx_fft = tf_fftshift2d(tf_fft2d(phs_only_cpx), [batch, num_channels, res_h, res_w])
    circ_filter = tf.convert_to_tensor(np_circ_filter(batch, num_channels, res_h, res_w, radius), dtype=tf.complex64)
    phs_only_cpx_fft_filtered = phs_only_cpx_fft * circ_filter

    # converts back to temporal domain
    phs_only_cpx_filtered = tf_ifft2d(tf_ifftshift2d(phs_only_cpx_fft_filtered, [batch, num_channels, res_h, res_w]))

    # propagate back to the center of the 3d volume
    if depth_shift != 0:
        tf_wavelength = tf.constant(np.array(wavelength).reshape(1,3,1,1))
        phs_only_cpx_filtered = propagator(phs_only_cpx_filtered, depth_shift) * tf_compl_exp(2*np.pi*depth_shift/tf_wavelength)

    # obtain amplitude and phs
    amp_filtered = tf.abs(phs_only_cpx_filtered)
    phs_filtered = tf.math.angle(phs_only_cpx_filtered)
    if normalize_output:
        phs_filtered = phs_filtered / 2.0 / np.pi + 0.5

    return amp_filtered, phs_filtered



class Propagation(abc.ABC):
    """ Base class for propagation, partially adapted from [Sitzmann et al. 2018]
    """
    def __init__(self,
                 input_shape,
                 pitch,
                 wavelengths,
                 double_pad):
        self.input_shape  = input_shape
        if double_pad:
            self.m_pad    = input_shape[0] // 2
            self.n_pad    = input_shape[1] // 2
        else:
            self.m_pad    = 0
            self.n_pad    = 0
        self.wavelengths  = wavelengths[None, :, None, None]
        self.wave_nos     = 2. * np.pi / wavelengths
        self.pitch        = pitch
        self.fx, self.fy  = self._tf_xy_grid()
        self.unit_phase_shift = None

    # tensorflow grid
    def _tf_xy_grid(self):
        M = tf.convert_to_tensor(self.input_shape[0] + 2 * self.m_pad, dtype=tf.float32)
        N = tf.convert_to_tensor(self.input_shape[1] + 2 * self.n_pad, dtype=tf.float32)

        x = tf.range(-(N // 2), N // 2, delta=1)
        y = tf.range(-(M // 2), M // 2, delta=1)
        [x, y] = tf.meshgrid(x, y)

        # Spatial frequency
        fx = x / (self.pitch * N)  # max frequency = 1/(2*pixel_size)
        fy = y / (self.pitch * M)

        fx = fx[tf.newaxis, tf.newaxis, :, :]
        fy = fy[tf.newaxis, tf.newaxis, :, :]      

        fx = tf_ifftshift2d(fx)
        fy = tf_ifftshift2d(fy)
  
        return fx, fy        

    @abc.abstractmethod
    def _unit_phase_shift(self):
        """Compute unit distance phase shift
        """

    def _propagate(self, input_field, z_dist):
        padded_input_field = tf.pad(input_field,
                                    [[0, 0], [0, 0], [self.m_pad, self.m_pad], [self.n_pad, self.n_pad]])

        H = tf_compl_exp(z_dist * self.unit_phase_shift, dtype=tf.complex64)

        obj_fft = tf_fft2d(padded_input_field)
        out_field = tf_ifft2d(obj_fft * H)

        return out_field[:, :, self.m_pad:out_field.shape[2]-self.m_pad, self.n_pad:out_field.shape[3]-self.n_pad]

    def __call__(self, input_field, z_dist):
        return self._propagate(input_field, z_dist)

# Fresnel approximation
class FresnelPropagation(Propagation):
    def __init__(self, 
                 input_shape,
                 pitch,
                 wavelengths,
                 double_pad):
        super(FresnelPropagation, self).__init__(input_shape, pitch, wavelengths, double_pad)
        self.unit_phase_shift = self._unit_phase_shift()

    def _unit_phase_shift(self):
        squared_sum = tf.square(self.fx) + tf.square(self.fy)
        phase_shift = -1. * self.wavelengths * np.pi * squared_sum
        return phase_shift


# angular spectrum propagation
class ASPropagation(Propagation):
    def __init__(self,
                 input_shape,
                 pitch,
                 wavelengths,
                 double_pad):
        super(ASPropagation, self).__init__(input_shape, pitch, wavelengths, double_pad)
        self.unit_phase_shift = self._unit_phase_shift()

    def _unit_phase_shift(self):
        phase_shift = 2 * (np.pi * (1 / self.wavelengths) * 
            tf.sqrt(1. - (self.wavelengths * self.fx) ** 2 - (self.wavelengths * self.fy) ** 2))
        return phase_shift


def tf_propagator(input_shape,
                  pitch,
                  wavelengths,
                  method = "as",
                  double_pad = False):
    switcher = {
        "as": ASPropagation,
        "fresnel": FresnelPropagation
    }
    propagator = switcher.get(method, "invalid method")
    if propagator == "invalid method":
        raise ValueError("invalid propgation method")
    return propagator(input_shape=input_shape,
                        pitch=pitch,
                        wavelengths=wavelengths,
                        double_pad=double_pad)