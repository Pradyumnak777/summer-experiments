from pathlib import Path

#everything is anchored to src/ so the pipeline resolves the same paths from any cwd
SRC_ROOT = Path(__file__).resolve().parent.parent

#raw per-cell tiffs live here as <subject_dir>/<cell_id>.tif with an optional <cell_id>mask.tif
RAW_DIR = SRC_ROOT / "raw_tiffs"

#dtype/axes corrected copies of the raw tiffs, these are the ones imagej can actually open
FIXED_DIR = SRC_ROOT / "data" / "singlecell_tiffs_fixed_new"

OUT_DIR_A = SRC_ROOT / "data" / "singlecell_chA_new"
OUT_DIR_B = SRC_ROOT / "data" / "singlecell_chB_new"
OUT_DIR_MASK = SRC_ROOT / "data" / "singlecell_mask_new"
OUT_DIR_PREVIEW = SRC_ROOT / "data" / "singlecell_previews_new"

#one playable movie per cell per channel, as opposed to the per frame pngs above. the frame rate is
#baked into the directory name because it is a guess rather than something read off the data (see
#VIDEO_FPS below), so re-encoding at a different rate lands beside the old pass instead of silently
#overwriting it and leaving you unable to tell the two apart
def fps_tag(fps):
    #7.0 -> "7" and 7.5 -> "7.5", keeps the common integer case clean
    return f"{fps:g}"


def video_out_dir(channel, fps, masked=False):
    """channel is "A", "B" or "mask" -> data/chA_videos_7, data/chA_videos_7_masked, ..."""
    stem = "mask" if channel == "mask" else f"ch{channel}"
    suffix = "_masked" if masked else ""
    return SRC_ROOT / "data" / f"{stem}_videos_{fps_tag(fps)}{suffix}"

#per-cell annotation/derived tiffs living in subject_dir subfolders (hand_drawn_mask, masked_img,
#output, etc.), mirrored here as png frames under the same subfolder names
ANNOTATIONS_OUT_DIR = SRC_ROOT / "data" / "annotations"

#each split lands next to its source dir as <source>_split/{train,validation}
SPLIT_SUFFIX = "_split"

CH_A_INDEX = 0
CH_B_INDEX = 1

LOWER_PCT = 0.35
UPPER_PCT = 99.65

CAPTION_A = "an image of gene burst in a microscopic cell"
CAPTION_B = "an image of condensate in a microscopic cell"
CAPTION_COMPOSITE = "an image of a cell with magenta condensate and green gene burst"

#kept identical to the old train_test_split.py so the partition stays reproducible
VAL_FRACTION = 0.05
SEED = 42

SAVE_PREVIEWS = True

#the raw tiffs carry no timing whatsoever, they were rewritten by tifffile and only kept a shape
#header, so there is no imagej finterval, no ome TimeIncrement and no exposure tag to read an
#acquisition interval back out of. the analysis output in raw_tiffs/*/output/*graphs.png plots its
#x axis as "Time [frames]" for the same reason. 7 fps is the rate the lab's own exported movies of
#this same 60 frame data use (bleach_corrected_*_MIP_merged*.mp4 sitting in src/), so that is the
#playback rate here, override it with --fps once the real interval is confirmed with shunli
VIDEO_FPS = 20

#same 128x128 the models actually train on (VAE_disent/data_utils.py resizes to this at load time),
#so what you watch is what the network sees. raw crops are 72-143 px and odd sized, which would also
#be unplayable as yuv420p since that needs even dimensions
VIDEO_SIZE = 128

#matched to VAE_disent/data_utils.py: bilinear for the image channels, nearest for the mask so it
#stays strictly binary and never grows soft edges
VIDEO_RESAMPLE = "bilinear"
MASK_RESAMPLE = "nearest"

#value the masked out background is set to. data_utils.py fills it with -1.0 in [-1,1] space, which
#is exactly 0 here in uint8 terms
MASK_FILL = 0

#visually lossless for this kind of sparse low res content while staying small on a full disk
VIDEO_CRF = 18

#masks are encoded truly losslessly instead. they are binary, so lossy encoding both softens the
#edges (~3% of pixels come back as neither 0 nor 255) and costs *more* bits than lossless does,
#since the quantiser spends them on ringing it invented. these mask videos are for looking at, the
#png masks under OUT_DIR_MASK stay the source of truth for anything that trains
MASK_CRF = 0

#each clip is only 60 tiny frames so encoding is dominated by process startup, running a handful in
#parallel turns an hour long pass into a couple of minutes. kept well under nproc because the box is
#shared with labmates
VIDEO_WORKERS = 8
