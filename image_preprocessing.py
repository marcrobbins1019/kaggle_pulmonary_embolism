import numpy as np
import cv2

def compress_dimensions(vol_3d,dimension):

    indices = np.linspace(0,vol_3d.shape[0] - 1, dimension[0])
    z_compressed = vol_3d[indices]
    z_compressed_xyz = np.swapaxes(z_compressed,0,2)
    xyz_compressed = cv2.resize(z_compressed_xyz,dsize=(dimension[1],dimension[2]))
    zyx_compressed = np.swapaxes(xyz_compressed,0,2)

    return zyx_compressed

def get_HU_window(vol_3d,window):

    window_min = window[1] - window[0]/2
    window_max = window[1] + window[0]/2

    clipped = np.clip(vol_3d,window_min,window_max)
    normalized = (clipped + clipped.min()) / (clipped.max() - clipped.min())
    return normalized


def get_representation(full_series, window, dimension, normalize_mm):

    vol_3d = full_series['volume_3d']

    compressed = compress_dimensions(vol_3d=vol_3d,dimension=dimension)
    final = get_HU_window(vol_3d=compressed,window=window)

    full_series['volume_3d'] = final
    return full_series





