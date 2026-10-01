import numpy as np
import tifffile


def infer_axes(ndim):
    if ndim == 4:
        return "TCYX"
    if ndim == 3:
        #masks legitimately have no channel axis, the caller decides whether 3d is a problem
        return "TYX"
    return None


def fix_dtype(arr):
    #raw exports come out as float64 which imagej cannot open, and the values are really integer
    #photon counts that got upcast upstream, so go back to uint16 when that is exact and fall back
    #to float32 (the widest float imagej reads) when it is not
    if np.issubdtype(arr.dtype, np.floating):
        whole_numbers = np.allclose(arr, np.round(arr))
        in_uint16_range = arr.min() >= 0 and arr.max() <= 65535
        if whole_numbers and in_uint16_range:
            return arr.astype(np.uint16)
        return arr.astype(np.float32)
    return arr


def load_and_fix(src_path, dst_path):
    """read a raw tiff, correct dtype and axes, write the imagej readable copy, return (arr, axes)"""
    arr = tifffile.imread(src_path)
    axes = infer_axes(arr.ndim)
    if axes is None:
        print(f"SKIP  {src_path}  (unexpected shape {arr.shape})")
        return None, None

    fixed = fix_dtype(arr)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    #the axes tag is what makes imagej read this back as a hyperstack rather than a flat page stack
    tifffile.imwrite(dst_path, fixed, imagej=True, metadata={"axes": axes})
    return fixed, axes
