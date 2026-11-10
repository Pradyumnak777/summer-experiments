'''
gene burst detection, ported from christina's geneburst_detection_new.py ("Christinas Scripts/"), so it can be imported
(her script runs everything at import, and has the spotiflow model as a global)

same functions and the same default parameters, only changes:
    - the spotiflow model is passed in (load_spot_model) instead of being a global, and loaded lazily
    - prints and plotting removed
    - the radius choice (her plot_compare/single_graph) is pick_radius, without the figure
see vae_viz.py mask_track() for how they're chained
'''
import math
import numpy as np
import scipy.optimize
from scipy import constants
from scipy.ndimage import binary_erosion, binary_fill_holes
from sklearn.metrics import r2_score

#her defaults. sizes/distances are in pixels of the image, variances in pixels², the rest unitless
PARAMETERS = {'min_r_square': 0.22, 'max_x_variance': 15, 'max_y_variance': 15, 'max_distance': 5, 'min_rel_intensity': 0.3,
              'min_signal_duration': 3, 'circle_radius': 3, 'fit_size': 15, 'integrated_intensity': True}
PEAK_THRESHOLD_REL = 0.35   #peak_local_max: a peak has to be >= 35% of the brightest pixel
PEAK_MIN_DISTANCE  = 20     #peak_local_max: px between peaks
SPOT_CACHE_DIR     = "/scratch/pbk5339/.cache/spotiflow"


def load_spot_model(cache_dir=SPOT_CACHE_DIR):
    #spotiflow's pretrained "general" model, on the cpu (the frames are tiny, and "auto" would pick cuda:0 on the shared box)
    from spotiflow.model import Spotiflow
    return Spotiflow.from_pretrained("general", map_location="cpu", cache_dir=cache_dir, verbose=False)


def extract_roi(image, x_center, y_center, size):
    roi_out_of_bounds = [False, False, False, False] # xmin, xmax, ymin, ymax
    #round size/2 up or down if size is uneven, to ensure the roi has a width and height equal to size
    x_min = round(x_center) - math.floor(size/2)
    x_max = round(x_center) + math.ceil(size/2)
    y_min = round(y_center) - math.floor(size/2)
    y_max = round(y_center) + math.ceil(size/2)
    if x_min < 0:
        x_min = 0
        roi_out_of_bounds[0] = True
    if y_min < 0:
        y_min = 0
        roi_out_of_bounds[2] = True
    if x_max >= image.shape[0]:
        x_max = image.shape[0] - 1
        roi_out_of_bounds[1] = True
    if y_max >= image.shape[1]:
        y_max = image.shape[1] - 1
        roi_out_of_bounds[3] = True
    x, y = np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1)
    XY = np.meshgrid(x, y)
    return [XY, image[x_min:x_max, y_min:y_max], roi_out_of_bounds]


def circularity(radius, area):
    #how well the mask covers a dot. 0.781 is a circle, 1 is squareish
    return (area*1.0 / (4 * radius * radius))


def gauss_2d(xy, x_0, y_0, x_variance, y_variance, amplitude, background):
    x, y = xy
    return amplitude * np.exp(-(((x-x_0)**2/(2*x_variance)) + ((y-y_0)**2/(2*y_variance)))) + background


def fit_image(image, peak, size):
    #fits a 2d gaussian in a size x size box around peak -> (params, r², background, roi)
    #params = [x_0, y_0, x_variance, y_variance, amplitude, background], x = row, y = col
    roi = extract_roi(image, peak[0], peak[1], size)
    xy = np.vstack((roi[0][0].ravel(), roi[0][1].ravel()))

    guess_x_min = guess_y_min = int(size/2) - int(size/6)
    guess_x_max = guess_y_max = int(size/2) + int(size/6) + 1

    guess_params = [peak[0], peak[1], 10, 10, np.max(roi[1]), np.mean(roi[1])]
    bounds = [[roi[0][0][0][guess_x_min], roi[0][1][guess_y_min][0], 0, 0, np.min(roi[1]), np.min(roi[1])],
              [roi[0][0][0][guess_x_max], roi[0][1][guess_y_max][0], np.inf, np.inf, 10*np.max(roi[1]), 10*np.max(roi[1])]]

    background = np.mean(roi[1])
    try:
        popt, pcov = scipy.optimize.curve_fit(gauss_2d, xy, roi[1].ravel(), p0=guess_params, bounds=bounds)
        r_square = r2_score(roi[1].ravel(), gauss_2d(xy, *popt).ravel())
    except RuntimeError:
        popt = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        r_square = 0
    return popt, r_square, background, roi[1]


def euc_distance(x1, x2):
    return np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)


def classify_bursts(fit_params, r_square, background, thresholds):
    #per frame: is there a real burst (positive_signal_mask) and where (spot_position), from the gaussian fits of that frame's peaks.
    #a burst has to pass every threshold, last min_signal_duration frames, and move <= max_distance px between frames
    burst = lambda params, r2, thresholds: r2 > thresholds['min_r_square'] and params[2] < thresholds['max_x_variance'] and params[3] < thresholds['max_y_variance'] and abs(params[4]/params[5]) > thresholds['min_rel_intensity']

    positive_signal_mask = np.zeros(len(fit_params), dtype=bool)
    intensity = np.zeros((len(fit_params)))
    fitted_params_true = []
    i = 0
    signal_active = False
    spot_position = np.zeros((len(fit_params), 2))
    spot_position[:, :] = np.nan
    for params_peaks, r2_peaks in zip(fit_params, r_square): #loop frames
        true_peaks = []
        true_positive = False
        for peak_ind, (params, r2) in enumerate(zip(params_peaks, r2_peaks)): #loop peaks in frame
            if burst(params, r2, thresholds):
                true_positive = True
                if signal_active == False:
                    signal_active = True
                    if i + thresholds['min_signal_duration'] > len(fit_params):
                        true_positive = False
                        signal_active = False
                        continue
                    for j in range(thresholds['min_signal_duration']):
                        False_peaks = 0
                        for k in range(len(fit_params[i+j])):
                            if not burst(fit_params[i+j][k], r_square[i+j][k], thresholds):
                                False_peaks += 1
                        if False_peaks == len(fit_params[i+j]):
                            true_positive = False
                            signal_active = False

            if true_positive:
                true_peaks.append(peak_ind)

        if len(true_peaks) == 1:
            params = params_peaks[true_peaks[0]]
            if i > 0 and positive_signal_mask[i-1] == True:
                distance_pre = euc_distance(spot_position[i-1], params[0:2])
            else:
                distance_pre = 0
            if distance_pre > thresholds['max_distance']:
                intensity[i] = background[i]
                positive_signal_mask[i] = False
                fitted_params_true.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            else:
                if not thresholds['integrated_intensity']:
                    intensity[i] = params[4]
                else:
                    intensity[i] = 2 * constants.pi * params[4] * np.sqrt(params[2] * params[3])
                positive_signal_mask[i] = True
                spot_position[i, 0] = params[0]
                spot_position[i, 1] = params[1]
                fitted_params_true.append(params)

        elif len(true_peaks) == 0:
            intensity[i] = background[i]
            positive_signal_mask[i] = False
            fitted_params_true.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

        elif len(true_peaks) > 1:
            #several real peaks: the one closest to the last detected spot, or the brightest if there's none yet
            closest_peak = 0
            distance = np.inf
            pre_frame = i-1
            while positive_signal_mask[pre_frame] == False and pre_frame >= 0:
                pre_frame -= 1
            if pre_frame == -1:
                peak_int = np.zeros(len(true_peaks))
                for j, peak_ind in enumerate(true_peaks):
                    peak_int[j] = params_peaks[peak_ind][4]
                for k, peak in enumerate(peak_int):
                    if peak == np.max(peak_int):
                        idx = k
                params = params_peaks[true_peaks[idx]]
                positive_signal_mask[i] = True
                spot_position[i, 0] = params[0]
                spot_position[i, 1] = params[1]
                intensity[i] = params[4]
                fitted_params_true.append(params)
            else:
                for j, peak_ind in enumerate(true_peaks):
                    if euc_distance(spot_position[pre_frame], params_peaks[peak_ind][0:2]) < distance:
                        distance = euc_distance(spot_position[pre_frame], params_peaks[peak_ind][0:2])
                        closest_peak = j
                if distance > thresholds['max_distance']:
                    positive_signal_mask[i] = False
                    fitted_params_true.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                else:
                    params = params_peaks[true_peaks[closest_peak]]
                    if not thresholds['integrated_intensity']:
                        intensity[i] = params[4]
                    else:
                        intensity[i] = 2 * constants.pi * params[4] * np.sqrt(params[2] * params[3])
                    positive_signal_mask[i] = True
                    spot_position[i, 0] = params[0]
                    spot_position[i, 1] = params[1]
                    fitted_params_true.append(params)

        i += 1

    return intensity, positive_signal_mask, spot_position, fitted_params_true


def classify_spots(spot_list):
    #spotiflow's per-frame spot (nan = none) -> burst yes/no + position
    n_frames = len(spot_list)
    positive_signal_mask = np.zeros(n_frames, dtype=bool)
    spot_position = np.full((n_frames, 2), np.nan)
    for i, spot in enumerate(spot_list):
        if not np.isnan(spot).any():
            positive_signal_mask[i] = True
            spot_position[i] = spot
    return positive_signal_mask, spot_position


def recursively_populate_mask(img, mask, threshold, x, y, visited, minX, maxX, minY, maxY):
    #flood fill from (x, y) over the pixels above threshold, inside the box [minX, maxX] x [minY, maxY]
    if x < 0 or x >= img.shape[1]:
        return
    if y < 0 or y >= img.shape[0]:
        return
    if (x < minX or x > maxX or y < minY or y > maxY):
        return
    elif (visited[y, x] != 0):
        return
    elif (img[y, x] <= threshold):
        visited[y, x] = 1
        return
    elif (img[y, x] > threshold):
        visited[y, x] = 1
        mask[y, x] = 1
        recursively_populate_mask(img, mask, threshold, x-1, y, visited, minX, maxX, minY, maxY)
        recursively_populate_mask(img, mask, threshold, x, y-1, visited, minX, maxX, minY, maxY)
        recursively_populate_mask(img, mask, threshold, x, y+1, visited, minX, maxX, minY, maxY)
        recursively_populate_mask(img, mask, threshold, x+1, y, visited, minX, maxX, minY, maxY)
    visited[y, x] = 1
    return


def dot_mask_from_mean_threshold(final_img, radius, frame_num, positive_signal_mask, spot_position):
    #the burst mask per frame: threshold = mean of the box of half-width radius around the spot, flood fill from the spot
    #over the pixels above it, fill holes. frames without a spot get an empty mask
    #NOTE: positive_signal_mask is changed in place for spots too close to the border (as in her script)
    dotmask_frames = []
    integrated_intensity = []
    masked_img_frames = []
    thresh_list = []
    areas_ratios = []
    for t in range(frame_num):
        image = final_img[t].copy()
        if positive_signal_mask[t] and np.all(np.isfinite(spot_position[t])):
            y, x = map(int, spot_position[t])
            roi = image[y-radius:y+radius, x-radius:x+radius]
            mask = np.zeros(image.shape, dtype=np.uint8)
            visited = np.zeros(image.shape, dtype=np.uint8)
            if roi.size == 0 or x-radius < 0 or y-radius < 0:
                positive_signal_mask[t] = False
                dotmask_frames.append(np.zeros_like(image))
                integrated_intensity.append(0)
                masked_img_frames.append(np.zeros_like(image))
                thresh_list.append(0)
                areas_ratios.append(0)
            else:
                thresh = roi.mean()
                recursively_populate_mask(image, mask, thresh, x, y, visited, x-radius, x+radius, y-radius, y+radius)
                mask = binary_fill_holes(mask)
                dotmask_frames.append(mask)
                integrated_intensity.append(image[mask > 0].sum())
                masked_img_frames.append(np.where((mask > 0) & (image > 0), image, 0))
                thresh_list.append(thresh)
                areas_ratios.append(circularity(radius, (mask > 0).sum()))
        else:
            dotmask_frames.append(np.zeros_like(image))
            integrated_intensity.append(0)
            masked_img_frames.append(np.zeros_like(image))
            thresh_list.append(0)
            areas_ratios.append(0)
    integrated_intensity = np.array(integrated_intensity)
    return dotmask_frames, integrated_intensity, masked_img_frames, np.array(thresh_list), areas_ratios


def filter_spots_by_mask(spots, mask):
    #keep only spots inside a 2d binary mask. spots expected as (N,2) or (2,N)
    spots = np.asarray(spots, dtype=float)
    if spots.size == 0:
        return np.empty((0, 2), dtype=float)
    if spots.ndim == 2 and spots.shape[0] == 2 and spots.shape[1] != 2:
        spots = spots.T
    H, W = mask.shape
    keep = []
    for s in spots:
        y = int(round(float(s[0])))
        x = int(round(float(s[1])))
        if 0 <= y < H and 0 <= x < W and mask[y, x] > 0:
            keep.append([float(s[0]), float(s[1])])
    if len(keep) == 0:
        return np.empty((0, 2), dtype=float)
    return np.asarray(keep, dtype=float)


def erode_mask(mask, px):
    #shrink a binary mask inward by px pixels, so spots on the hard cell-mask edge get dropped
    if px <= 0:
        return mask.astype(bool)
    return binary_erosion(mask.astype(bool), iterations=int(px))


def getSpotLoc(model, frames, img, mask_stack, radius, gaussian_locations, positive_signal_mask, search_mask=None):
    #spotiflow's spot per frame (nan = none). with several spots: the one closest to the gaussian spot if that frame
    #has one, else the brightest (within 10 px of the previous spot, if there is one)
    #search_mask: optional [H, W] binary region (not in her script), spots outside it are dropped before one is picked
    spot_prob_thresh = 0.3   #stricter than spotiflow's default, drops low confidence noise spots
    mask_erosion_px = 4      #spots within this many px of the cell edge are dropped

    #global normalization using masked pixels only
    valid_pixels = img[mask_stack > 0]
    if valid_pixels.size > 0:
        global_min = valid_pixels.min()
        global_max = valid_pixels.max()
    else:
        global_min = img.min()
        global_max = img.max()

    spot_list = np.zeros((frames, 2))
    spot_list[:, :] = np.nan
    for t in range(frames):
        frame = img[t].astype(np.float32)
        mask = mask_stack[t]
        frame_masked = frame.copy()
        frame_masked[mask == 0] = 0
        frame_norm = (frame_masked - global_min) / (global_max - global_min + 1e-8)
        frame_norm[mask == 0] = 0
        pred = model.predict(frame_norm, prob_thresh=spot_prob_thresh, verbose=False)
        spots = pred[0] if isinstance(pred, tuple) else pred
        mask_eroded = erode_mask(mask, mask_erosion_px)
        spots = filter_spots_by_mask(spots, mask_eroded)
        if search_mask is not None:
            spots = filter_spots_by_mask(spots, search_mask)   #not eroded, the region is small enough already
        if len(spots) == 0:
            spot_list[t] = np.array([np.nan, np.nan])
        else:
            #the most recent previous valid spot
            prev_frame = t - 1
            while prev_frame >= 0:
                if np.isnan(spot_list[prev_frame]).any():
                    prev_frame -= 1
                else:
                    break
            if prev_frame < 0:
                if positive_signal_mask[t]:
                    distances = np.linalg.norm(spots - gaussian_locations[t], axis=1)
                    spot_list[t] = spots[np.argmin(distances)]
                else:
                    best_idx = 0
                    best_intensity = -np.inf
                    for j, spot in enumerate(spots):
                        y = int(round(spot[0]))
                        x = int(round(spot[1]))
                        y0 = max(0, y - radius)
                        y1 = min(frame.shape[0], y + radius + 1)
                        x0 = max(0, x - radius)
                        x1 = min(frame.shape[1], x + radius + 1)
                        intensity = frame[y0:y1, x0:x1].max()
                        if intensity > best_intensity:
                            best_intensity = intensity
                            best_idx = j
                    spot_list[t] = spots[best_idx]
            else:
                if positive_signal_mask[t]:
                    distances = np.linalg.norm(spots - gaussian_locations[t], axis=1)
                    spot_list[t] = spots[np.argmin(distances)]
                else:
                    prev_spot = spot_list[prev_frame]
                    max_distance = 10  #pixels
                    best_idx = None
                    best_intensity = -np.inf
                    for j, spot in enumerate(spots):
                        distance = np.linalg.norm(spot - prev_spot)
                        if distance > max_distance:
                            continue
                        y = int(round(spot[0]))
                        x = int(round(spot[1]))
                        y0 = max(0, y - radius)
                        y1 = min(frame.shape[0], y + radius + 1)
                        x0 = max(0, x - radius)
                        x1 = min(frame.shape[1], x + radius + 1)
                        intensity = frame[y0:y1, x0:x1].max()
                        if intensity > best_intensity:
                            best_intensity = intensity
                            best_idx = j
                    if best_idx is not None:
                        spot_list[t] = spots[best_idx]
                    else:
                        spot_list[t] = np.array([np.nan, np.nan])
    return spot_list


def pick_radius(threshes, integrated_intensities, positive_signal_mask, radii):
    #her plot_compare/single_graph without the figure: per radius, fit a line of integrated intensity vs threshold over
    #the frames with a (gaussian) burst, keep the radius with the highest r². 0 if no radius gets a fit (< 2 burst frames)
    best_radius, max_rsquare = 0, 0
    for r in radii:
        x, y = threshes[r][positive_signal_mask], integrated_intensities[r][positive_signal_mask]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        #no line through < 2 points, or through points that all share one x (eg every threshold 0, as no spotiflow spot
        #got a mask there): polyfit divides by the x spread and fails, and r² means nothing then anyway
        if len(x) < 2 or np.ptp(x) == 0:
            continue
        m, b = np.polyfit(x, y, 1)
        ss_res = np.sum((y - (m*x + b))**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        if r_squared > max_rsquare:
            max_rsquare, best_radius = r_squared, r
    return best_radius
