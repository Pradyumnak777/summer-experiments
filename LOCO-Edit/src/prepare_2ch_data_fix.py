import tifffile
import numpy as np

IN_PATH  = "single_cell_files/20260611_HP_bh1-PP7_15_TIGRE-PCP-mSG_17-01_processed-Orthogonal Projection-146_bc/0.tif"
OUT_PATH = "0_test_uint16.tif"

data = tifffile.imread(IN_PATH)
print("original:", data.shape, data.dtype, "min", data.min(), "max", data.max())

#just changing dtype here, not rescaling values. round to nearest int and clip to what uint16 can hold
if data.max() > 65535 or data.min() < 0:
    print("warning: values fall outside 0-65535, clipping will lose info, this test wont be clean")

data_u16 = np.clip(np.round(data), 0, 65535).astype(np.uint16)
print("converted:", data_u16.shape, data_u16.dtype, "min", data_u16.min(), "max", data_u16.max())

#data is already shaped (T, C, Y, X) so tifffile can write it straight back as a hyperstack
tifffile.imwrite(OUT_PATH, data_u16, imagej=True, metadata={'axes': 'TCYX'})
print(f"saved to {OUT_PATH}")