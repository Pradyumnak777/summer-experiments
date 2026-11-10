#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:49:20 2024

@author: manyu
"""
#data scatterplot
import numpy as np
import glob
import os
import re
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
from skimage import io,morphology,measure
from skimage.measure import regionprops_table
from skimage.segmentation import clear_border
from skimage.draw import disk
from skimage.draw import circle_perimeter
from cellpose import models
from cellpose.io import imread
from spotiflow.model import Spotiflow
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation
import imageio
import math
from skimage.feature import peak_local_max
from matplotlib.patches import Rectangle

#from aicsimageio import AICSImage
import czifile as cz
#import xml.etree.ElementTree as ET
import scipy as sp

#from skimage import measure
import scipy.io as sio

from sklearn.metrics import r2_score
import scipy.optimize
from scipy import constants
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_fill_holes

###shape properties########
import shapely.geometry as sg
import shapely.ops as so
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

######for spatial clustering#######
import sklearn
from sklearn.cluster import HDBSCAN

###########write tiff file######
import tifffile
import pandas as pd
#plt.close('all')

parameters={'min_r_square':0.22, 'max_x_variance':15, 'max_y_variance':15, 'max_distance':5, 'min_rel_intensity':0.3, 'min_signal_duration':3,'circle_radius': 3, 'fit_size':15,'integrated_intensity':True}

############################## local functions #################################
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
    XY = np.meshgrid(x,y)
    return [XY, image[x_min:x_max, y_min:y_max], roi_out_of_bounds]

#another way to measure how well the mask covers a dot.
#0.781 is circle
#1 is squareish
def circularity(radius, area):
    return (area*1.0 /( 4* radius * radius ))

def gauss_2d(xy, x_0, y_0, x_variance, y_variance, amplitude, background):
        """
        2D Gaussian function.
        """
        x, y = xy
        return amplitude * np.exp(-(((x-x_0)**2/(2*x_variance)) + ((y-y_0)**2/(2*y_variance)))) + background

def fit_image(image,peak,size):
    #image=image.transpose()
    #row, col = peak #added by Jonila
    #x_center, y_center = col, row #added by Jonila
    roi = extract_roi(image, peak[0], peak[1], size)
    #roi = extract_roi(image, y_center, x_center, size) #added by Jonila
    xy = np.vstack((roi[0][0].ravel(), roi[0][1].ravel()))
    # fig=plt.figure()
    # plt.imshow(roi[1])
       
    guess_x_min = guess_y_min = int(size/2) - int(size/6)
    guess_x_max = guess_y_max = int(size/2) + int(size/6) + 1
    
    guess_params = [peak[0], peak[1], 10, 10, np.max(roi[1]), np.mean(roi[1])]
    #guess_params = [x_center, y_center, 10, 10, np.max(roi[1]), np.mean(roi[1])] #added by Jonila
         #guess_params= [x_guess[guess_frame_index], y_guess[guess_frame_index], 1, 1, np.max(roi[1]), np.mean(roi[1])]
    bounds = [[roi[0][0][0][guess_x_min], roi[0][1][guess_y_min][0], 0, 0, np.min(roi[1]), np.min(roi[1])], [roi[0][0][0][guess_x_max], roi[0][1][guess_y_max][0], np.inf, np.inf, 10*np.max(roi[1]), 10*np.max(roi[1])]]

    background = np.mean(roi[1])
    try:
        popt, pcov = scipy.optimize.curve_fit(gauss_2d, xy, roi[1].ravel(), p0=guess_params, bounds=bounds)
        r_square = r2_score(roi[1].ravel(), gauss_2d(xy, *popt).ravel())
        

    except RuntimeError:
        #print('Fit was not successfull. Setting fit_params and r_square to np.nan and 0!')
        popt= [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        r_square = 0
       
    return popt, r_square, background, roi[1]

def plot_compare(n_frames,thresh2,thresh3,thresh4,classified_intensities, classified_intensities2, classified_intensities3, classified_intensities4, path_name): #cw
    out_path = "compare"
    # 1. FIXED LAYOUT: Use subplots_adjust, NOT tight_layout
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5, wspace=0.5)
    fig.patch.set_facecolor('white')    

    #axes[0,0].set_title('Gaussian fit')  
    axes[0,0].set_title('Scatter 3')
    axes[0,1].set_title('Scatter 4')
    axes[0,2].set_title('Scatter 5')
    axes[1,0].set_title('Radius 3')
    axes[1,1].set_title('Radius 4')
    axes[1,2].set_title('Radius 5')
      
    axes[0,0].set_xlabel('mean intens val 3x3')
    axes[0,0].set_ylabel('Radius 3 intensity')
    axes[0,1].set_xlabel('mean intens val 4x4')
    axes[0,1].set_ylabel('Radius 4 intensity')    
    axes[0,2].set_xlabel('mean intens val 5x5')
    axes[0,2].set_ylabel('Radius 5 intensity')    
    axes[1,0].set_xlabel('Time [frames]')
    axes[1,0].set_ylabel('Intensity [a.u.]')
    axes[1,1].set_xlabel('Time [frames]')
    axes[1,1].set_ylabel('Intensity [a.u.]')
    axes[1,2].set_xlabel('Time [frames]')
    axes[1,2].set_ylabel('Intensity [a.u.]')
    axes[2,0].set_xlabel('gaussian')
    axes[2,0].set_ylabel('Radius 3 intensity')
    axes[2,1].set_xlabel('gaussian')
    axes[2,1].set_ylabel('Radius 4 intensity')    
    axes[2,2].set_xlabel('gaussian')
    axes[2,2].set_ylabel('Radius 5 intensity')    
    
    # classified_intensities2 = [x / 3 for x in classified_intensities2]
    # classified_intensities3 = [x / 4 for x in classified_intensities3]
    # classified_intensities4 = [x / 5 for x in classified_intensities4]
    
    axes[1,0].set_xlim(0, n_frames)
    axes[1,0].set_ylim(np.nanmin(classified_intensities2) * 0.9, np.nanmax(classified_intensities2) * 1.1) 
    axes[1,1].set_xlim(0, n_frames)
    axes[1,1].set_ylim(np.nanmin(classified_intensities3) * 0.9, np.nanmax(classified_intensities3) * 1.1) 
    axes[1,2].set_xlim(0, n_frames)
    axes[1,2].set_ylim(np.nanmin(classified_intensities4) * 0.9, np.nanmax(classified_intensities4) * 1.1) 
    n_frames = len(classified_intensities2)
    classified_intensities = classified_intensities[positive_signal_mask]
    # classified_intensities2 = classified_intensities2[positive_signal_mask]
    # classified_intensities3 = classified_intensities3[positive_signal_mask]
    # classified_intensities4 = classified_intensities4[positive_signal_mask]
    ln, = axes[1,0].plot(np.arange(n_frames), classified_intensities2[:n_frames], color='black')
    ln2, = axes[1,1].plot(np.arange(n_frames), classified_intensities3[:n_frames], color='black')
    ln3, = axes[1,2].plot(np.arange(n_frames), classified_intensities4[:n_frames], color='black')

    best_rsquared_radius = 0
    max_rsquare=0.6 # anything below this value i will consider useless and we wont use it for graph 
    thresh2p = thresh2[positive_signal_mask]
    thresh3p = thresh3[positive_signal_mask]
    thresh4p = thresh4[positive_signal_mask]
    classified_intensitiesp2 = classified_intensities2[positive_signal_mask]
    classified_intensitiesp3 = classified_intensities3[positive_signal_mask]
    classified_intensitiesp4 = classified_intensities4[positive_signal_mask]
    axes[0,0].scatter(thresh2p,classified_intensitiesp2) 
    axes[0,1].scatter(thresh3p,classified_intensitiesp3) 
    axes[0,2].scatter(thresh4p,classified_intensitiesp4)
    axes[2,0].scatter(classified_intensities,classified_intensitiesp2) 
    axes[2,1].scatter(classified_intensities,classified_intensitiesp3) 
    axes[2,2].scatter(classified_intensities,classified_intensitiesp4) 
    m, b = np.polyfit(thresh2p, classified_intensitiesp2, 1)
    y_pred = m*thresh2p + b
    ss_res = np.sum((classified_intensitiesp2 - y_pred)**2)
    ss_tot = np.sum((classified_intensitiesp2 - np.mean(classified_intensitiesp2))**2)
    r_squared = 1 - ss_res/ss_tot
    if r_squared>max_rsquare:
        max_rsquare = r_squared
        best_rsquared_radius=3
    idx = np.argsort(thresh2p)
    axes[0,0].plot(thresh2p[idx], y_pred[idx])
    #print("r_squared: ",r_squared)
    axes[0,0].text(
        0.05, 0.95,
        f"R² = {r_squared:.3f}",
        transform=axes[0,0].transAxes,
        va='top'
    )
    m, b = np.polyfit(thresh3p, classified_intensitiesp3, 1)
    y_pred = m * thresh3p + b
    ss_res = np.sum((classified_intensitiesp3 - y_pred)**2)
    ss_tot = np.sum((classified_intensitiesp3 - np.mean(classified_intensitiesp3))**2)
    r_squared = 1 - ss_res/ss_tot
    if r_squared>max_rsquare:
        max_rsquare = r_squared
        best_rsquared_radius=4
    idx = np.argsort(thresh3p)
    axes[0,1].plot(thresh3p[idx], y_pred[idx])
    #print("r_squared: ",r_squared)
    axes[0,1].text(
        0.05, 0.95,
        f"R² = {r_squared:.3f}",
        transform=axes[0,1].transAxes,
        va='top'
    )
    m, b = np.polyfit(thresh4p, classified_intensitiesp4, 1)
    y_pred = m * thresh4p + b
    ss_res = np.sum((classified_intensitiesp4 - y_pred)**2)
    ss_tot = np.sum((classified_intensitiesp4 - np.mean(classified_intensitiesp4))**2)
    r_squared = 1 - ss_res/ss_tot
    if r_squared>max_rsquare:
        max_rsquare = r_squared
        best_rsquared_radius=5
    idx = np.argsort(thresh4p)
    axes[0,2].plot(thresh4p[idx], y_pred[idx])
    #print("r_squared: ",r_squared)
    axes[0,2].text(
        0.05, 0.95,
        f"R² = {r_squared:.3f}",
        transform=axes[0,2].transAxes,
        va='top'
    )
    m, b = np.polyfit(classified_intensities, classified_intensitiesp2, 1)
    y_pred = m*classified_intensities + b
    ss_res = np.sum((classified_intensitiesp2 - y_pred)**2)
    ss_tot = np.sum((classified_intensitiesp2 - np.mean(classified_intensitiesp2))**2)
    r_squared = 1 - ss_res/ss_tot
    if r_squared>max_rsquare:
        max_rsquare = r_squared
        best_rsquared_radius=3
    idx = np.argsort(classified_intensities)
    axes[2,0].plot(classified_intensities[idx], y_pred[idx])
    #print("r_squared: ",r_squared)
    axes[2,0].text(
        0.05, 0.95,
        f"R² = {r_squared:.3f}",
        transform=axes[2,0].transAxes,
        va='top'
    )
    m, b = np.polyfit(classified_intensities, classified_intensitiesp3, 1)
    y_pred = m * classified_intensities + b
    ss_res = np.sum((classified_intensitiesp3 - y_pred)**2)
    ss_tot = np.sum((classified_intensitiesp3 - np.mean(classified_intensitiesp3))**2)
    r_squared = 1 - ss_res/ss_tot
    if r_squared>max_rsquare:
        max_rsquare = r_squared
        best_rsquared_radius=4
    idx = np.argsort(classified_intensities)
    axes[2,1].plot(classified_intensities[idx], y_pred[idx])
    #print("r_squared: ",r_squared)
    axes[2,1].text(
        0.05, 0.95,
        f"R² = {r_squared:.3f}",
        transform=axes[2,1].transAxes,
        va='top'
    )
    m, b = np.polyfit(classified_intensities, classified_intensitiesp4, 1)
    y_pred = m * classified_intensities + b
    ss_res = np.sum((classified_intensitiesp4 - y_pred)**2)
    ss_tot = np.sum((classified_intensitiesp4 - np.mean(classified_intensitiesp4))**2)
    r_squared = 1 - ss_res/ss_tot
    if r_squared>max_rsquare:
        max_rsquare = r_squared
        best_rsquared_radius=5
    idx = np.argsort(classified_intensities)
    axes[2,2].plot(classified_intensities[idx], y_pred[idx])
    #print("r_squared: ",r_squared)
    axes[2,2].text(
        0.05, 0.95,
        f"R² = {r_squared:.3f}",
        transform=axes[2,2].transAxes,
        va='top'
    )
    
    return fig, axes, best_rsquared_radius

def plot_fitted_movie(orig_data, fit_params, classified_intensities, classified_intensities2,classified_intensities3,classified_intensities4,mask2,mask3,mask4,save=False, out_path=''):
    """
    Export an animation of the raw data and the Gaussian fit without black glitches.
    """
    n_frames = min(len(classified_intensities),len(classified_intensities2),len(classified_intensities3),len(classified_intensities4))
    x_size, y_size = orig_data[0].shape
    X = np.arange(0, x_size + 1)
    Y = np.arange(0, y_size + 1)
    XY = np.meshgrid(Y, X)

    # 1. FIXED LAYOUT: Use subplots_adjust, NOT tight_layout
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)
    fig.patch.set_facecolor('white')

    # Setup Axes

    # 2. LOCK LIMITS: Calculate global min/max to prevent flickering/shifting
    all_raw_min = min(frame.min() for frame in orig_data)
    all_raw_max = max(frame.max() for frame in orig_data)
    # Estimate fit max to ensure consistent colormap
    all_fit_max = max(gauss_2d(XY, *p).max() for p in fit_params)
    vmin, vmax = all_raw_min, max(all_raw_max, all_fit_max)

    # Lock the intensity plot limits so it doesn't "jump"


    frames_artists = []
    for t in range(n_frames):
        # 3. BACKGROUND CLEARING: Add fig.patch to every frame to fix black margins
        bg_rect = Rectangle((0, 0), 1, 4, transform=fig.transFigure, 
                        facecolor='white', zorder=-10)
        fig.add_artist(bg_rect) # Add it to the figure
        
        # axes[0,0].invert_yaxis()
        # axes[0,1].invert_yaxis()
        #axes[1,0].axis('off') # Hide the empty bottom-left plot

        for ax in [axes[0,0], axes[0,1], axes[1,1], axes[1,0], axes[0,2], axes[0,3],axes[1,2],axes[1,3]]:
            ax.set_facecolor('white')
        
        axes[0,0].set_aspect('equal')
        axes[0,1].set_aspect('equal')
        axes[0,2].set_aspect('equal')
        axes[0,3].set_aspect('equal')
        axes[0,4].set_aspect('equal')
        axes[1,0].spines['top'].set_visible(False)
        axes[1,0].spines['right'].set_visible(False)
        axes[1,1].spines['top'].set_visible(False)
        axes[1,1].spines['right'].set_visible(False)

        axes[0,0].set_title('Gaussian fit')
        axes[0,1].set_title('Radius 3')
        axes[0,2].set_title('Radius 4')
        axes[0,3].set_title('Radius 5')
        axes[0,4].set_title('original')
        
        for i in range(4):           
            axes[1,i].set_xlabel('Time [frames]')
            axes[1,i].set_ylabel('Intensity [a.u.]')
            
        axes[1,0].set_xlim(0, n_frames)
        axes[1,0].set_ylim(np.nanmin(classified_intensities) * 0.9, np.nanmax(classified_intensities) * 1.1) 
        axes[1,1].set_xlim(0, n_frames)
        axes[1,1].set_ylim(np.nanmin(classified_intensities2) * 0.9, np.nanmax(classified_intensities2) * 1.1) 
        axes[1,2].set_xlim(0, n_frames)
        axes[1,2].set_ylim(np.nanmin(classified_intensities3) * 0.9, np.nanmax(classified_intensities3) * 1.1) 
        axes[1,3].set_xlim(0, n_frames)
        axes[1,3].set_ylim(np.nanmin(classified_intensities4) * 0.9, np.nanmax(classified_intensities4) * 1.1) 

        # 4. DATA PLOTTING: Use fixed vmin/vmax for both pcolormesh calls
        y0, x0, xvar, yvar, amp, bg = fit_params[t]
        fit_frame = gauss_2d(XY, x_0=x0, y_0=y0, x_variance=yvar, 
                             y_variance=xvar, amplitude=amp, background=bg)
        
        im1 = axes[0,0].pcolormesh(XY[0], XY[1], fit_frame, vmin=vmin, vmax=vmax, cmap='Greys_r')
        im2 = axes[0,1].pcolormesh(XY[0], XY[1], mask2[t], 
                                  vmin=vmin, vmax=vmax, cmap='Greys_r')
        im3 = axes[0,2].pcolormesh(XY[0], XY[1], mask3[t], 
                                  vmin=vmin, vmax=vmax, cmap='Greys_r')
        im4 = axes[0,3].pcolormesh(XY[0], XY[1], mask4[t], 
                                  vmin=vmin, vmax=vmax, cmap='Greys_r')
        im5 = axes[0,4].pcolormesh(XY[0], XY[1], orig_data[t], 
                                  vmin=vmin, vmax=vmax, cmap='Greys_r')
        
        
        # 5. LINE PLOTTING: Note the comma after 'ln' for proper artist grabbing
        # classified_intensities = classified_intensities[positive_signal_mask]
        # classified_intensities2 = classified_intensities2[positive_signal_mask]
        # classified_intensities3 = classified_intensities3[positive_signal_mask]
        # classified_intensities4 = classified_intensities4[positive_signal_mask]
        n = min(t + 1, len(classified_intensities))
        n = min(n, len(classified_intensities2))
        ln, = axes[1,0].plot(np.arange(n), classified_intensities[:n], color='black')
        ln2, = axes[1,1].plot(np.arange(n), classified_intensities2[:n], color='black')
        ln3, = axes[1,2].plot(np.arange(n), classified_intensities3[:n], color='black')
        ln4, = axes[1,3].plot(np.arange(n), classified_intensities4[:n], color='black')
        
        # Append all artists, including the background patch
        frames_artists.append([bg_rect, im1, im2, im3, im4, im5, ln, ln2, ln3, ln4])

    # 6. SAVE SETTINGS: blit=False is mandatory for PillowWriter artifacts
    ani = animation.ArtistAnimation(fig, frames_artists, interval=300, blit=False)

    if save and out_path:
        # Use metadata disposal to ensure each frame replaces the previous one
        ani.save(out_path, writer='pillow', fps=3, 
                 savefig_kwargs={'facecolor': 'white'},
                 metadata={'disposal': 2})
        print(f"Animation saved to {out_path}")

    #plt.show()
    return ani
    
def euc_distance(x1,x2):
    return np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)

def classify_bursts(fit_params, r_square, background, thresholds):
    """
    Classify each frame of a movie into wether a true signal or just background signal is present, given the parameters of a Gaussian fit.
    """
    
    burst = lambda params, r2, thresholds: r2 > thresholds['min_r_square'] and params[2] < thresholds['max_x_variance'] and params[3] < thresholds['max_y_variance'] and abs(params[4]/params[5]) > thresholds['min_rel_intensity']

    positive_signal_mask = np.zeros(len(fit_params), dtype=bool)
    intensity = np.zeros((len(fit_params)))
    fitted_params_true=[]
    i = 0
    signal_active = False
    spot_position = np.zeros((len(fit_params),2))
    spot_position[:,:]=np.nan
    for params_peaks, r2_peaks in zip(fit_params, r_square): #loop frames
        true_peaks = []
        true_positive=False
        for peak_ind, (params,r2) in enumerate(zip(params_peaks, r2_peaks)): #loop peaks in frame
            #if not np.isnan(params[0]):
                # figure=plt.figure()
                # x_size,y_size=thresholds['fit_size'],thresholds['fit_size']
                # x,y=np.arange(params[0]-x_size,params[0]+x_size,1),np.arange(params[0]-y_size,params[0]+y_size,1)
                # XY=np.array(np.meshgrid(y,x))
                # print(i)
                # print(params)
                # plt.pcolormesh(XY[0], XY[1], gauss_2d(XY, *params), cmap='Greys_r')
                # plt.show()
                #plt.clf()
            #print(i)
            #print(burst(params, r2, thresholds))
            #print(params,r2)
            if burst(params, r2, thresholds):
                true_positive = True
                if signal_active == False:
                    signal_active = True
                    if i + thresholds['min_signal_duration'] > len(fit_params):
                        true_positive = False
                        signal_active = False
                        continue
                    for j in range(thresholds['min_signal_duration']):
                        False_peaks=0
                        for k in range(len(fit_params[i+j])):  
                            if not burst(fit_params[i+j][k], r_square[i+j][k], thresholds):
                                False_peaks+=1
                        if False_peaks==len(fit_params[i+j]):
                            true_positive = False
                            signal_active = False
                            
            if true_positive:
                true_peaks.append(peak_ind)
    
        #print(true_peaks)
        if len(true_peaks) == 1:
            params=params_peaks[true_peaks[0]]
            if i>0 and positive_signal_mask[i-1]==True:
                distance_pre=euc_distance(spot_position[i-1], params[0:2])
            else:
                distance_pre=0
            if distance_pre>thresholds['max_distance']:
                signal_active == False
                intensity[i] = background[i]
                positive_signal_mask[i] = False
                fitted_params_true.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                
            else:
                if not thresholds['integrated_intensity']:
                    intensity[i] = params[4]
                else:
                    intensity[i] = 2 * constants.pi * params[4] * np.sqrt(params[2] * params[3])
                positive_signal_mask[i] = True
                spot_position[i,0]=params[0]
                spot_position[i,1]=params[1]
                fitted_params_true.append(params)
            
        elif len(true_peaks) == 0:
            signal_active == False
            intensity[i] = background[i]
            positive_signal_mask[i] = False
            fitted_params_true.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            
        elif len(true_peaks) > 1:
            #TODO: think about how to deal with multiple detected peaks
            #just select the highest intensity in the first frame is not working very well.
            #print("Warning: More than 1 true spot detected in frame " + str(i) + ".")
            closest_peak=0
            distance=np.inf
            # peak_initial=0
            # idx=0
            pre_frame=i-1
            while positive_signal_mask[pre_frame]==False and pre_frame>=0:
                pre_frame-=1
            #print(params_peaks)
            if pre_frame==-1:
                peak_int=np.zeros(len(true_peaks))
                for j,peak_ind in enumerate(true_peaks):
                    peak_int[j] = params_peaks[peak_ind][4]
                #print(peak_int)
                for k,peak in enumerate(peak_int):
                    if peak==np.max(peak_int):
                        idx=k
                #print(idx)
                params=params_peaks[true_peaks[idx]] 
                positive_signal_mask[i] = True
                spot_position[i,0]=params[0]
                spot_position[i,1]=params[1]
                intensity[i]=params[4]
                fitted_params_true.append(params)
            else:
                #print('current frame: ' + str(i))
                #print(pre_frame)
                for j,peak_ind in enumerate(true_peaks):
                    #print(j)
                    # print(pre_frame)
                    # print(true_peaks[j])
                    # print(params_peaks[true_peaks[j]])
                    #print(params_peaks[peak_ind][0:2])
                    #print(spot_position[pre_frame])
                    #print(euc_distance(spot_position[pre_frame], params_peaks[peak_ind][0:2]))
                    if euc_distance(spot_position[pre_frame], params_peaks[peak_ind][0:2])<distance:
                        distance=euc_distance(spot_position[pre_frame], params_peaks[peak_ind][0:2])
                        closest_peak=j
                    #print('closests: ' + str(closest_peak))
                if distance>thresholds['max_distance']:
                    intensity[i] == np.nan
                    positive_signal_mask[i] = False
                    fitted_params_true.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                else:
                    params=params_peaks[true_peaks[closest_peak]]
                    if not thresholds['integrated_intensity']:
                        intensity[i] = params[4]
                    else:
                        intensity[i] = 2 * constants.pi * params[4] * np.sqrt(params[2] * params[3])
                    positive_signal_mask[i] = True
                    spot_position[i,0]=params[0]
                    spot_position[i,1]=params[1]
                    #print(spot_position[i,0:2])
                    fitted_params_true.append(params)
        
        i += 1
        
    return intensity, positive_signal_mask,spot_position,fitted_params_true

def classify_spots(spot_list): #classify bursts but with adrianas spot movement code aadded by cw
    n_frames = len(spot_list)

    positive_signal_mask = np.zeros(n_frames, dtype=bool)
    spot_position = np.full((n_frames, 2), np.nan)

    for i, spot in enumerate(spot_list):

        if np.isnan(spot).any():
            # No detection
            positive_signal_mask[i] = False

        else:
            # Spot detected
            positive_signal_mask[i] = True
            spot_position[i] = spot

    return positive_signal_mask, spot_position

def dot_mask_from_mean_threshold(final_img,radius,frame_num,positive_signal_mask,spot_position):
    dotmask_frames = []
    integrated_intensity = []
    masked_img_frames = []
    thresh_list = []
    areas_ratios = []
    for t in range(frame_num): #do the circle thing for all frames
        image = final_img[t].copy()       
        if positive_signal_mask[t] and np.all(np.isfinite(spot_position[t])):
            y, x = map(int, spot_position[t])        
            roi = image[y-radius:y+radius, x-radius:x+radius]
            mask = np.zeros(image.shape, dtype=np.uint8)
            visited = np.zeros(image.shape, dtype=np.uint8)
            if roi.size == 0 or x-radius <0 or y-radius <0:
                positive_signal_mask[t] = False
                dotmask_frames.append(np.zeros_like(image))
                integrated_intensity.append(0)
                masked_img_frames.append(np.zeros_like(image))
                thresh_list.append(0)
                areas_ratios.append(0)  
            else:
                thresh = roi.mean()
                #mask[y-radius:y+radius, x-radius:x+radius] = roi > thresh
                recursively_populate_mask(image, mask, thresh, x, y, visited, x-radius, x+radius, y-radius, y+radius)
                mask = binary_fill_holes(mask)
                dotmask_frames.append(mask)
                integrated_intensity.append(image[mask > 0].sum())
                masked_img_frames.append(np.where((mask > 0) & (image > 0), image, 0))
                thresh_list.append(thresh)     
                areas_ratios.append(circularity(radius,(mask>0).sum()))  
                print("square_val: ",4*radius*radius)
                print("actual_area:",(mask>0).sum())
                print("circularity: ",circularity(radius,(mask>0).sum()))
        else:
            dotmask_frames.append(np.zeros_like(image))
            integrated_intensity.append(0)
            masked_img_frames.append(np.zeros_like(image))
            thresh_list.append(0)  
            areas_ratios.append(0)  
    integrated_intensity = np.array(integrated_intensity)
    return dotmask_frames,integrated_intensity,masked_img_frames, np.array(thresh_list), areas_ratios

def recursively_populate_mask(img, mask, threshold, x, y, visited, minX, maxX, minY, maxY):
    if x < 0 or x >= img.shape[1]:
        return
    if y < 0 or y >= img.shape[0]:
        return    
    if(x<minX or x>maxX or y<minY or y>maxY):
        return 
    elif(visited[y,x]!=0):
        return
    elif(img[y,x]<=threshold):
        visited[y,x]=1
        return
    elif(img[y,x]>threshold):
        #print("this code is running", x, y)
        visited[y,x]=1
        mask[y,x] = 1
        recursively_populate_mask(img, mask, threshold, x-1, y, visited, minX, maxX, minY, maxY)
        recursively_populate_mask(img, mask, threshold, x, y-1, visited, minX, maxX, minY, maxY)
        recursively_populate_mask(img, mask, threshold, x, y+1, visited, minX, maxX, minY, maxY)
        recursively_populate_mask(img, mask, threshold, x+1, y, visited, minX, maxX, minY, maxY)
    visited[y,x]=1
    return

def filter_spots_by_mask(spots, mask): #adriana
    """
    Keep only spots inside a 2D binary mask.
    spots expected as (N,2) or (2,N).
    """
    spots = np.asarray(spots, dtype=float)
    if spots.size == 0:
        return np.empty((0, 2), dtype=float)
    # handle (2, N) -> (N, 2)
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

def create_excel(masks, image, PATHSAVE, raw_path, radius, cellnum):
    all_frames=[]
    for i in range(len(masks)):
        cond_props = regionprops_table(masks[i].astype(np.uint8), intensity_image=image[i], properties=[
            'area',
            'perimeter',
            'orientation',
            'eccentricity',
            'major_axis_length',
            'minor_axis_length',
            'centroid'
        ])
        df = pd.DataFrame(cond_props)
        df = df.rename(columns={
            'centroid-1': 'centroidX',
            'centroid-0': 'centroidY'
        })
        df['cell'] = cellnum   # add time/frame column
        df['radius']=radius
        df['frame'] = i   # add time/frame column
        all_frames.append(df)
    # combine everything into one table
    final_df = pd.concat(all_frames, ignore_index=True)
    fname = PATHSAVE + "datas.csv"
    if not os.path.exists(fname):
        final_df.to_csv(fname, mode="w", index=False)  # write header
    else:
        final_df.to_csv(fname, mode="a", index=False, header=False)
    return 

def output_cell_intensity(intensities, cell_num, rad):
    fname = "/Users/mxd451/Desktop/Single Cells/20260611_HP_bh1-PP7_TIGRE-PCP-mSG_13-01_processed-Orthogonal Projection-142_bc/Intensities.csv"

    # Ensure 1D numeric list (prevents nested-array bugs)
    intensities = np.asarray(intensities).ravel().tolist()

    # Row: cell ID + values
    row = [f"Cell_{cell_num}_rad_{rad}"] + intensities

    # Column names must match EXACT length of row
    columns = ["Cell"] + list(range(len(intensities)))

    # Create DataFrame
    df = pd.DataFrame([row], columns=columns)

    # Write or append
    if not os.path.exists(fname):
        df.to_csv(fname, index=False)
    else:
        df.to_csv(fname, mode="a", index=False, header=False)

def output_data(filename, data):
    fname = filename
    data = np.asarray(data).ravel().tolist()
    df = pd.DataFrame([data])
    if not os.path.exists(fname):
        df.to_csv(fname, index=False)
    else:
        df.to_csv(fname, mode="w", index=False, header=False)

def erode_mask(mask, px): #adriana
    """
    Shrink a binary mask inward by `px` pixels.


    Why: spot detection runs on a frame where everything outside the cell mask
    has been zeroed out (see frame_masked below). That hard cutout creates an
    artificial intensity edge right at the cell boundary, and Spotiflow will
    sometimes fire a "spot" on that edge itself. A naive mask test
    (filter_spots_by_mask against the raw mask) still keeps those edge
    detections, since they sit exactly on/just inside the boundary. Filtering
    against an eroded mask instead drops anything within `px` pixels of the
    boundary, which is enough to remove edge artifacts while keeping genuine
    interior spots.
    """
    if px <= 0:
        return mask.astype(bool)
    return binary_erosion(mask.astype(bool), iterations=int(px))

def getSpotLoc(frames, img, mask_stack, fit_params, radius, gaussian_locations, positive_signal_mask):  #adriana
    # Stricter Spotiflow detection probability threshold (0-1). Leave as None to
    # use the model's default.
    #
    # Why this is set instead of None: channel_1 (the single-dot channel) has a
    # much lower raw dynamic range than channel_0 in this dataset (e.g. ~300-1700
    # vs channel_0's tens of thousands, sometimes saturating at 65535). Each file
    # is independently min/max-normalized to [0,1] before detection (see
    # frame_norm below), so in a low-dynamic-range file that normalization
    # stretches mostly camera noise across the same range a real spot would
    # occupy elsewhere — Spotiflow then fires on several noise bumps, which can
    # make a single-dot channel look like a multi-dot one. Raising the confidence
    # threshold disproportionately removes those low-confidence noise detections
    # while leaving genuine, high-confidence spots (which channel_0's real puncta
    # should produce) intact. Lower this back toward None/0.3 if real spots in a
    # dim channel start getting dropped instead.
    spot_prob_thresh = 0.5
    # -------------------------
    # Spot-detection tuning: keep only spots that are really inside the cell
    # -------------------------
    # Pixels within this many pixels of the cell mask boundary are excluded from
    # the final spot list. The hard mask cutout used to zero out background below
    # creates an artificial intensity edge at the cell boundary, which Spotiflow
    # can mistake for a real spot. Eroding the mask before filtering removes those
    # boundary artifacts. Raised from 2 -> 4 after still seeing spots right at/
    # outside the nucleus edge with the smaller margin. Raise further if edge
    # false-positives still get through; lower it (e.g. back to 2, or to 0) if
    # real spots near the membrane start getting dropped instead.
    mask_erosion_px = 4
    # -------------------------
    # Global normalization using masked pixels only
    # -------------------------
    valid_pixels = img[mask_stack > 0]
    if valid_pixels.size > 0:
        global_min = valid_pixels.min()
        global_max = valid_pixels.max()
    else:
        global_min = img.min()
        global_max = img.max()
    overlay_stack = []  
    # -------------------------
    # frame loop
    # ------------------------- 
    spot_list = np.zeros((frames,2))
    spot_list[:,:]=np.nan
    for t in range(frames):
        frame = img[t].astype(np.float32)
        mask = mask_stack[t]
        # apply mask before normalization
        frame_masked = frame.copy()
        frame_masked[mask == 0] = 0
        # normalize
        frame_norm = (frame_masked - global_min) / (global_max - global_min + 1e-8)
        frame_norm[mask == 0] = 0
        # predict
        pred = model.predict(frame_norm, prob_thresh=spot_prob_thresh)
        spots = pred[0] if isinstance(pred, tuple) else pred
        # filter to spots inside the cell, eroded inward so boundary artifacts
        # from the hard mask cutout above don't get counted as real spots
        mask_eroded = erode_mask(mask, mask_erosion_px)
        spots = filter_spots_by_mask(spots, mask_eroded)
        if len(spots) == 0:
            spot_list[t] = np.array([np.nan, np.nan])
        # elif len(spots) == 1:
        #     spot_list[t] = spots[0]
        # Multiple spots
        else:
            # Find the most recent previous valid spot
            prev_frame = t - 1
            while prev_frame >= 0:
                if np.isnan(spot_list[prev_frame]).any():
                    prev_frame -= 1
                else:
                    break
            # If no previous spot exists, use the highest intensity spot
            if prev_frame < 0:
                #find gaussian centroid and choose nearest to it if it exists:
                if (positive_signal_mask[t]):

                    # Compute Euclidean distance from every candidate
                    distances = np.linalg.norm(spots - gaussian_locations[t], axis=1)
                    # Pick the closest
                    closest_idx = np.argmin(distances)
                    spot_list[t] = spots[closest_idx]
                else :
                    #pick highest intensity
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
                #find gaussian centroid and choose nearest to it if it exists:
                if (positive_signal_mask[t]):

                    # Compute Euclidean distance from every candidate
                    distances = np.linalg.norm(spots - gaussian_locations[t], axis=1)
                    # Pick the closest
                    closest_idx = np.argmin(distances)
                    spot_list[t] = spots[closest_idx]
                else:
                    prev_spot = spot_list[prev_frame]

                    max_distance = 10  # pixels (adjust as needed)

                    best_idx = None
                    best_intensity = -np.inf

                    for j, spot in enumerate(spots):

                        # Distance from previous spot
                        distance = np.linalg.norm(spot - prev_spot)

                        # Skip spots that are too far away
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

                    # If at least one nearby spot was found
                    if best_idx is not None:
                        spot_list[t] = spots[best_idx]
                    else:
                        #No nearby spot found
                        #spot_list[t] = np.array([np.nan, np.nan])
                        #pick highest intensity
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
                    # #or just pick the nearest to your previous spot.
                    # prev_spot = spot_list[prev_frame]

                    # # Compute Euclidean distance from every candidate
                    # distances = np.linalg.norm(spots - prev_spot, axis=1)

                    # # Pick the closest
                    # closest_idx = np.argmin(distances)
                    # spot_list[t] = spots[closest_idx]
                    spots = []
    return spot_list
#################################################################################


############################# end local functions ###############################
#################################################################################


# ################################ setting up cellpose: ################################
# ######################################################################################
# ###### estimated diameter for segmentation. For large datasets its better to look at one image first and make sure it works.
# ###### otherwise, set to False and t will be estimated for each image. This will take time
idiameter=100 #for 24h Olaparib #120 #60 for nuclei >>>>> CRITICAL PARAMETER

#estimated_diameter=200 #for control and 5h >>>>> CRITICAL PARAMETER
#model_type='cyto' or 'nuclei' or 'cyto2'
modelname="cyto2"
#model = models.CellposeModel(model_type=modelname)
model = Spotiflow.from_pretrained("general")
#model = models.Cellpose(model_type='cyto2')#for for control and 5h

# if NUCLEUS channel does not exist, set the second channel to 0
channels = [[0,0]] # IF YOU HAVE GRAYSCALE
# ###################################################################################### cell time x, y, sumint, aspect ratio, major minor axis, cellpose, measure regionprops, maskarea, 
# ######################################################################################
###### number of columns and rows for the figure
nc=3;nr=1
###### saturation level for images
pci=.99 ##0.99
###### Figure layaout sizes
fact=5.7;fw=4;fh=1.5
###### filter images for segmentation (gausian)

sigma=2 #for control 
#sigma=10 #for 5h
#sigma=20 #24h
###### colormap parametersname = os.path.basename(rawcells[i]).replace('.tif', '')
minshift=.9
start = 0.3
stop = minshift


# folder='/Users/mxd451/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Du Lab/Jonila Shehu/script_data/Sox2pr_second batch data/'
# PATHSAVE='/Users/mxd451/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Du Lab/Jonila Shehu/script_data/Sox2pr_second batch data/'
folder='/Users/mxd451/Desktop/Single Cells/20260611_HP_bh1-PP7_TIGRE-PCP-mSG_13-01_processed-Orthogonal Projection-142_bc/'
PATHSAVE='/Users/mxd451/Desktop/Single Cells/20260611_HP_bh1-PP7_TIGRE-PCP-mSG_13-01_processed-Orthogonal Projection-142_bc/'
CASENAME=folder.split("/")[-2]

ZX=[]
ZY=[]
NEXP=0

finalcells=[]
all_tifs = glob.glob(os.path.join(folder, "*.tif"))
all_tifs = [os.path.normpath(f) for f in all_tifs]
raw_files = [os.path.normpath (f) for f in all_tifs if "mask" not in os.path.basename(f)]
mask_files = [os.path.normpath (f) for f in all_tifs if "mask" in os.path.basename(f)]
best_intensities = [] #best radius will be 0 if we cant find anything above R^2 = 0.7
chosen_mask = []

for raw_path in raw_files:
    raw_path = os.path.normpath(raw_path)
    raw_name = os.path.basename(raw_path)
    #print("Raw file:", raw_name)
    match = re.match(r'^(\d+)', raw_name)
    if not match:
        #print(f"Skipping file with no number: {raw_name}")
        continue
    whichcell = match.group(1)

    matching_mask = None
    for m in mask_files:
        mask_name = os.path.basename(m)
        if mask_name == f"{whichcell}mask.tif":
            matching_mask = m
            break

    if not matching_mask:
        #print(f"No mask found for: {raw_name}")
        continue

    raw_img = tifffile.imread(raw_path)
    raw_img = raw_img[:,0,:,:] #channel 0 only
    mask_img = tifffile.imread(matching_mask)

    if raw_img.shape != mask_img.shape:
        #print(f"Shape mismatch: {raw_name} vs {os.path.basename(matching_mask)}")
        continue
    #print(f"Matched: {os.path.basename(raw_path)} with {os.path.basename(matching_mask)}")
    
    final_img = raw_img * mask_img
    frame_num = final_img.shape[0]
    # Ncell = []
    # dot_intensities = []
    # bkg_intensities = []
    # frame_nums = []
    parameter_fit = [[] for _ in range(frame_num)]
    r_square      = [[] for _ in range(frame_num)]
    bkg_int       = np.zeros(frame_num, dtype=float)

    roi_ori       = []
    MS2_peaks_all = []
    
    summary_filename = f"cell_{whichcell}summary.gif"
    summary_path = os.path.join(PATHSAVE, summary_filename)
    
    frame_num = final_img.shape[0] 
    
    for frame in range(frame_num):
        frame_img = final_img[frame,:,:]
       
        peaks = peak_local_max(frame_img, exclude_border=False, threshold_rel=0.35, min_distance=20)
        for j,peak in enumerate(peaks):
            try:
                popt, r2, bkg, roi = fit_image(frame_img, peak, parameters['fit_size'])
                if r2 >= parameters['min_r_square']:
                    parameter_fit[frame].append(popt)
                    r_square[frame].append(r2)
                    roi_ori.append(roi)
            except Exception:
                continue
    intensity, positive_signal_mask,spot_position,fitted_params_true=classify_bursts(parameter_fit, r_square, bkg_int, parameters)
    overlay_frames = []
    radius = parameters['circle_radius']
    positive_signal_mask2, spot_position2 = classify_spots(getSpotLoc(frame_num,raw_img,mask_img,parameters, radius, spot_position, positive_signal_mask))

# find the max value in the entire movie so we know what “bright” is
    movie_max = final_img.max()

    for t in range(frame_num): #do the circle thing for all frames
        frame = final_img[t].copy()            
        if positive_signal_mask[t]:
            y0, x0 = map(int, spot_position[t])        # ensure integer coords
            rr, cc = circle_perimeter(y0, x0, radius, shape=frame.shape)
            frame[rr, cc] = movie_max 
        overlay_frames.append(frame)
        


# save to your PATHSAVE folder
    os.makedirs(PATHSAVE, exist_ok=True)
    fname    = os.path.basename(raw_path).replace('.tif','dotmask.tif')
    out_path = os.path.join(PATHSAVE, fname)
    tifffile.imwrite(out_path, overlay_frames)
    
        
    #cw
    
    #for i in range (3,6):
    dotmask_frames2, integrated_intensity2, masked_image2, threshes2, area_ratio2 = dot_mask_from_mean_threshold(final_img,3,frame_num,positive_signal_mask2,spot_position2)
    #fname2 = os.path.basename(PATHSAVE) + "mask_radius" + str(i) +".tif"
    fname2 = os.path.basename(raw_path).replace('.tif',"mask_radius" + str(3) +".tif")
    out_path = os.path.join(PATHSAVE, fname2)
    tifffile.imwrite(out_path, dotmask_frames2) #threshold spread CW
    fname2 = os.path.basename(raw_path).replace('.tif',"masked_radius" + str(3) +".tif")
    out_path = os.path.join(PATHSAVE, fname2)
    tifffile.imwrite(out_path, masked_image2) #threshold spread CW

    dotmask_frames3, integrated_intensity3, masked_image3, threshes3, area_ratio3 = dot_mask_from_mean_threshold(final_img,4,frame_num,positive_signal_mask2,spot_position2)
    #fname2 = os.path.basename(PATHSAVE) + "mask_radius" + str(i) +".tif"
    fname2 = os.path.basename(raw_path).replace('.tif',"mask_radius" + str(4) +".tif")
    out_path = os.path.join(PATHSAVE, fname2)
    tifffile.imwrite(out_path, dotmask_frames3) #threshold spread CW
    fname2 = os.path.basename(raw_path).replace('.tif',"masked_radius" + str(3) +".tif")
    out_path = os.path.join(PATHSAVE, fname2)
    tifffile.imwrite(out_path, masked_image3) #threshold spread CW

    dotmask_frames4, integrated_intensity4, masked_image4, threshes4, area_ratio4 = dot_mask_from_mean_threshold(final_img,5,frame_num,positive_signal_mask2,spot_position2)
    #fname2 = os.path.basename(PATHSAVE) + "mask_radius" + str(i) +".tif"
    fname2 = os.path.basename(raw_path).replace('.tif',"mask_radius" + str(5) +".tif")
    out_path = os.path.join(PATHSAVE, fname2)
    tifffile.imwrite(out_path, dotmask_frames4) #threshold spread CW
    fname2 = os.path.basename(raw_path).replace('.tif',"masked_radius" + str(3) +".tif")
    out_path = os.path.join(PATHSAVE, fname2)
    tifffile.imwrite(out_path, masked_image4) #threshold spread CW
    
    output_data("area3.csv", area_ratio2)
    output_data("area4.csv", area_ratio3)
    output_data("area5.csv", area_ratio4)
    
    final_plot, axes, best_radius = plot_compare(frame_num,threshes2, threshes3, threshes4, intensity, integrated_intensity2, integrated_intensity3, integrated_intensity4,"")
    #print("cell number: ",raw_path)
    #print("best radius ",best_radius)
    fname2 = os.path.basename(raw_path).replace('.tif',"graphs" +".png")
    out_path = os.path.join(PATHSAVE, fname2)
    
    #final_plot.canvas.draw()
    #image = np.array(final_plot.canvas.renderer.buffer_rgba())
    #tifffile.imwrite(out_path, final_plot) #threshold spread CW
    
    final_plot.savefig(out_path, dpi=300, bbox_inches='tight')
    #cw
    
    
    frame_list = [final_img[t] for t in range(frame_num)]
    movie_path = os.path.join(PATHSAVE,os.path.basename(raw_path).replace('.tif','_summary.gif'))
    plot_fitted_movie(
    orig_data              = frame_list,
    fit_params             = fitted_params_true,
    classified_intensities = intensity,
    classified_intensities2 = integrated_intensity2,
    classified_intensities3= integrated_intensity3,
    classified_intensities4= integrated_intensity4,
    mask2= masked_image2,
    mask3= masked_image3,
    mask4= masked_image4,
    save                   = True,
    out_path               = summary_path)
    # output_cell_intensity(integrated_intensity2,whichcell,3)
    # output_cell_intensity(integrated_intensity3,whichcell,4)
    # output_cell_intensity(integrated_intensity4,whichcell,5)
    #cw
    if best_radius == 3:
        best_intensities = integrated_intensity2
        chosen_mask = masked_image2
    elif best_radius == 4:
        best_intensities = integrated_intensity3
        chosen_mask = masked_image3
    elif best_radius ==5:
        best_intensities = integrated_intensity4
        chosen_mask = masked_image4
    if (best_radius!=0):
        create_excel(chosen_mask, final_img, PATHSAVE, raw_path, best_radius, raw_name)
    #cw
    
    #taken from jonias 
    if(best_radius!=0):
        output_cell_intensity(best_intensities,whichcell,best_radius)
        # d = {'frame': np.arange(len(best_intensities)),'intensity': best_intensities}
        # for key, value in d.items():
        #     print(key, np.shape(value))
        # df = pd.DataFrame(data=d)
        # name = os.path.basename(raw_path).replace('.tif', '')
        # save_path = os.path.join(PATHSAVE, f"{name}_output.xlsx")
        # df.to_excel(save_path, index=False)
    #end