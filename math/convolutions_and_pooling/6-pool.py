#!/usr/bin/env python3
'''
    a function def
    pool(images, kernel_shape, pool_shape, mode='max'):
    that performs a pooling on images:
    mode: max or avg
'''

import numpy as np

def pool(images, kernel_shape, stride, mode='max'):
    '''
        images: numpy.ndarray with shape (m, h, w, c)
            m: number of images
            h: height in pixels
            w: width in pixels
            c: number of channels
        kernel_shape: tuple of (kh, kw)
            kh: height of the kernel
            kw: width of the kernel
        stride: tuple of (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image
        mode: max or avg
        Returns: numpy.ndarray containing the pooled images
      '''
    m, h, w, c = images.shape  # Unpack the dimensions of the images
    kh, kw = kernel_shape      # Unpack the kernel dimensions
    sh, sw = stride            # Unpack the stride values

    # Calculate the output dimensions after pooling
    new_h = (h - kh) // sh + 1
    new_w = (w - kw) // sw + 1

    # Initialize the output array with the appropriate shape
    pooled = np.zeros((m, new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            # Extract the current slice of the input image
            image_slice = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

            if mode == 'max':
                # Perform max pooling
                pooled[:, i, j, :] = np.max(image_slice, axis=(1, 2))
            elif mode == 'avg':
                # Perform average pooling
                pooled[:, i, j, :] = np.mean(image_slice, axis=(1, 2))

    return pooled

