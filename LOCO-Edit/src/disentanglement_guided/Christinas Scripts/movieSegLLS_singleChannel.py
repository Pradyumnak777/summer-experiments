#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:16:09 2024

@author: manyu
"""

import numpy as np
import glob
import os
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
from skimage import io,morphology,measure
from skimage.segmentation import clear_border
from cellpose import models
from cellpose.io import imread
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as animation
import imageio
import math

#from aicsimageio import AICSImage
import czifile as cz
#import xml.etree.ElementTree as ET
import scipy as sp

#from skimage import measure
import scipy.io as sio

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


#plt.close('all')



############################## local functions #################################
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
model = models.CellposeModel(model_type=modelname)
#model = models.Cellpose(model_type='cyto2')#for for control and 5h

# if NUCLEUS channel does not exist, set the second channel to 0
channels = [[0,0]] # IF YOU HAVE GRAYSCALE
# ######################################################################################
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
###### colormap parameters
minshift=.9
start = 0.3
stop = minshift
#plt.imshow()2d

folder='/Users/clwang/Desktop/psu/Collaboration_Qunhua_Huijuan_Particle Tracking/Data/Rawdata'
PATHSAVE='/Users/clwang/Desktop/psu/Collaboration_Qunhua_Huijuan_Particle Tracking/Data/output'
CASENAME=folder.split("/")[-2]

#PATTERN=["DPHM_Nanog_MCP-01_MIP"] 
#PATTERN=["HM_Sox2_146Sox2pr-BSD-PP7_PCP-mSG-01_MIP_bc"] 
#PATTERN=["HM_Sox2_146Soxpr-BSD-PP7_PCP-mSG-01_MIP_bc"] 
#PATTERN=["HP_bh1_PP7_15_TIGRE_PCP-mSG_17-01_processed-Orthogonal Projection-86_bc"] 
PATTERN=["test"] 
#DIAMETER=[100,100]


###### get dimentions of dataset
ZX=[]
ZY=[]
NEXP=0
print("test")
for pattern in PATTERN:
    #fname=glob.glob(folder + "*" +pattern+'*.czi')
    fname=glob.glob(folder + "/" +pattern+'.tif')
    print(f"file name {pattern}")
    
    for i in range(0,len(fname)):#len(fname)
        filepath_sample=fname[i]
        print(f"reading image {filepath_sample}")
        # reading image
        #img = cz.imread(filepath_sample)
        img=tifffile.imread(filepath_sample)
        # getting the image from the structure
        #IM_cond=img[:,0,:,:,0]
        #IM_cond=img[0,:,0,0,:,:,0]
        #IM_cond=img[:,:,:]
        IM_cond=img[:,0,:,:]
        
        
        #frame_num = np.shape(IM_cond)[0]
        frame_num=10 #make it faster for test
        #x,y=np.arange(0,np.shape(IM_cond)[1]+1), np.arange(0,np.shape(IM_cond)[2]+1)
        #XY=np.array(np.meshgrid(y,x))
        #IM_MS2=img[:,0,:,:]
        #IM_MS2=img[:,:,:]
        #writer = imageio.get_writer(PATHSAVE + pattern +'test.tif')
        fig=plt.figure()
        #fig,ax=plt.subplots(2,1)
        plots = []
        clevels=[]
        Ncell=[]
        Mask_allf=[]
        IM_sum=np.zeros(np.shape(IM_cond)[1:3])
        for frame in range(0,frame_num): #read images and add up everything for final centroid
            print(f"executing frame {frame}")
            IM=IM_cond[frame,:,:]
            #IM=img[frame,:,:]
            IM_sum=IM_sum+IM
            #Mask_allf.append()
        #ZX=np.append(ZX,np.shape(IM)[0])
        #ZY=np.append(ZY,np.shape(IM)[1])
        print("before IM_sum eval", flush=True)
        MASK_sum, flows, styles = model.eval(IM_sum, diameter=idiameter, channels=channels)
        print("after IM_sum eval", flush=True)
        MASK_sum=clear_border(MASK_sum)
        print("after clear_border", flush=True)
        TMP_sum=measure.regionprops_table(MASK_sum,IM_sum,properties=('label','bbox','centroid_weighted'))
        print("after regionprops", flush=True)
        cnt_sum_x=TMP_sum['centroid_weighted-1']
        cnt_sum_y=TMP_sum['centroid_weighted-0']
        #plt.show()
        
        #plt.scatter(cnt_sum_x, cnt_sum_y, marker="*") 
        
        # #######################plot the mask_sum##################################
        # clevels=np.unique(MASK_sum)
        # plt.imshow(IM_sum,vmax=np.quantile(IM_sum.flatten(),pci),cmap='gray')
        # plt.contour(MASK_sum,linewidths=0.5,levels=clevels,colors="r")#COLORP
        # #########################################################
        ct_x=[]
        ct_y=[]
        cell_diameter=[]
        label=[]
        mask_mov=np.zeros((frame_num,IM_cond.shape[1],IM_cond.shape[2]))
        #mask_mov=np.zeros((frame_num,img.shape[1],img.shape[2]))
        
        #frame_num=10
        for frame in range(0,frame_num): #go back to every frame and using the x y from the cluster center, find individual centroids
            #print(f"2 executing frame {frame}")
            IM=IM_cond[frame,:,:]
            #IM=img[frame,:,:]
            IMG = sp.ndimage.gaussian_filter(IM, sigma, mode='constant')
            ##### performe cellpose masking with defult values for thresholds
            print(f"Starting Cellpose frame {frame}", flush=True)
            MASK, flows, styles= model.eval(IMG, diameter=idiameter, channels=channels)
            print(f"Finished Cellpose frame {frame}", flush=True)
            clevels=np.unique(MASK)            
            Ncell=len(clevels)-1
            print(Ncell)
            TMP=measure.regionprops_table(MASK,IM,properties=('label','bbox','centroid_weighted','axis_major_length'))
            cnt_x=TMP['centroid_weighted-1']
            cnt_y=TMP['centroid_weighted-0']
            celldiameter=TMP['axis_major_length']
            
            #plt.show()
            
            
            #plt.scatter(cnt_x, cnt_y) 
            mask_mov[frame]=MASK
            label.append(TMP['label'])
            ct_x.append(cnt_x)
            ct_y.append(cnt_y) # append centroid
            cell_diameter.append(celldiameter)
            print("flush", flush=True) 
        frames=[]
        #tifffile.imwrite(PATHSAVE+'test'+'mask.tif', mask_mov, bigtiff=True,planarconfig='separate')
        print("line 193", flush=True)
        
        for i,x in enumerate(ct_x): #clustering for each centroid
            print(x)
            frames.append([i for j in range(len(x))])
        frames=np.concatenate(frames).ravel()
        label=np.concatenate(label).ravel()
        ct_x=np.concatenate(ct_x).ravel()
        ct_y=np.concatenate(ct_y).ravel()
        cell_diameter=np.concatenate(cell_diameter).ravel()

            
        ct_xy=np.stack((ct_x,ct_y)).transpose()
        clusters=HDBSCAN(min_cluster_size=frame_num,max_cluster_size=frame_num,min_samples=10) #only count if have all frames
        clusters.fit(ct_xy)
        
        print("line 209")
        ##################filter out noise###############
        for clusteri in np.unique(clusters):
            if clusteri==-1 or clusteri==-2 or clusteri==-3:
                continue
            ind=np.argwhere(clusters.labels_==clusteri)
            f=frames[ind]
            u,c=np.unique(f,return_counts=True)
            if np.any(c!=1):
                clusters.labels_[ind]=-4
                
                
        ##################crop movies around cells############### 
       # cropsize=[]
            #print(np.unique(clusters.labels_))
        print("line 233")
        for clusteri in np.unique(clusters.labels_): # crop images
           # print(clusteri)
            if clusteri==-1 or clusteri==-2 or clusteri==-3 or clusteri==-4:
                cropsize=0
            else:
                ind=np.argwhere(clusters.labels_==clusteri)
                #print(clusters.labels_)
                #print(cell_diameter[ind])               
                cropsize=math.ceil(np.max(cell_diameter[ind]))+20
                #print(cropsize)
                #print(type(frame_num))
                #region=np.zeros((frame_num,1,cropsize,cropsize))
                region=np.zeros((frame_num,cropsize,cropsize))
                nuclei_mask=np.zeros((frame_num,cropsize,cropsize))
                for j in range(frame_num):#crop
                    print("starting crop")
                    #print(ct_x[ind][j][0])
                    current_label=label[ind][j]
                    #print(label[ind])
                    xmin=round(ct_x[ind][j][0])-math.floor(cropsize/2)
                    xmax=round(ct_x[ind][j][0])+math.ceil(cropsize/2)
                    ymin=round(ct_y[ind][j][0])-math.floor(cropsize/2)
                    ymax=round(ct_y[ind][j][0])+math.ceil(cropsize/2)
                    #print(ct_y[ind][j])
                    
                    if xmin<0:
                        xmin=0
                    if ymin<0:
                        ymin=0
                    # if xmax>=img.shape[2]:
                    #     xmax=img.shape[2]-1
                    # if ymax>=img.shape[1]:
                    #     ymax=img.shape[1]-1  
                    if xmax>=IM_cond.shape[2]:
                        xmax=IM_cond.shape[2]-1
                    if ymax>=IM_cond.shape[1]:
                        ymax=IM_cond.shape[1]-1  
                    #print(xmin,xmax,ymin,ymax)
                    #IM=img[0,j,0,0,ymin:ymax,xmin:xmax] #im masks 
                    #IM=img[j,ymin:ymax,xmin:xmax] 
                    IM=img[j,0,ymin:ymax,xmin:xmax] 
                    #print(mask_mov.shape)
                    IM_mask=mask_mov[j,ymin:ymax,xmin:xmax].copy()
                    # if clusteri==83:
                    #     fig=plt.figure()
                    #     plt.imshow(IM_mask!=current_label)
                    
                    IM_mask[IM_mask!=current_label]=0 #TODO: FIX THIS
                    
                    # if clusteri==83:
                    #     fig=plt.figure()
                    #     plt.imshow(IM_mask)
                    IM_mask[IM_mask!=0]=1 #TODO: FIX THIS
                    print(img.shape)
                    #print(MASK.shape)
                    print("IM shape = ",IM.shape)
                    print("region shape = ",region.shape)
                    #region[j,:,0:ymax-ymin,0:xmax-xmin]=IM 
                    region[j,0:ymax-ymin,0:xmax-xmin]=IM #0,135,1033 into 135 135
                    nuclei_mask[j,0:ymax-ymin,0:xmax-xmin]=IM_mask
                print(f"Saving cluster {clusteri}")
                print(region.shape)
                print(nuclei_mask.shape)    
                savefile = os.path.join(PATHSAVE, f"{clusteri}.tif")
                maskfile = os.path.join(PATHSAVE, f"{clusteri}_mask.tif")
                print("Cluster labels:", np.unique(clusters.labels_))
                print("PATHSAVE:", PATHSAVE)
                print("Output exists:", os.path.exists(PATHSAVE))
                tifffile.imwrite(savefile, region)
                tifffile.imwrite(maskfile, nuclei_mask)
                # tifffile.imwrite(PATHSAVE+str(clusteri)+'.tif', region, bigtiff=True,planarconfig='separate')
                # tifffile.imwrite(PATHSAVE+str(clusteri)+'mask.tif', nuclei_mask, bigtiff=True,planarconfig='separate')
                
                print("end")
                    
                    
                
            
            
        
            
#         maxi=np.max(clusters.labels_)
#         clusters.labels_[clusters.labels_==-1]=maxi+10
#         clusters.labels_[clusters.labels_==-4]=maxi+11
#         color=plt.get_cmap('Dark2')((clusters.labels_+1)%8)
#         color[clusters.labels_==112]=[0,0,0,1]
#         color[clusters.labels_==113]=[1,0,0,1]
       
        
#         plt.scatter(ct_xy[:,0],ct_xy[:,1], c=color)
        
     
#         #print(cnt_x)
#         #print(np.shape(ct_y))
            
#         #plt.scatter(ct_x, ct_y)    
            
#         NEXP=NEXP+1
# #DX=int(min(ZX)/2)
# #DY=int(min(ZY)/2)


#MATP=np.zeros(shape=(NEXP,int(DX*2-2)))
#MATP=np.zeros(shape=(NEXP,935))

















          #MASK=clear_border(MASK)
          #plots.append([plt.plot(frame,Ncell)])
          #writer.append_data(MASK)
          #plots.append([plt.pcolormesh(XY[0],XY[1],IM,animated=True)])
          #writer.append_data(MASK)
          #plots.append([plt.imshow(IM)])
          #plots.append([ax[0].imshow(IM),ax[1].imshow(IM)])
          #plots.append(plt.plot(range(0,frame), range(0,frame)))
      # interval = 100
      # #fps=10
      # # writer=FFMpegWriter(fps=fps)
      # ani = animation.ArtistAnimation(fig, plots, interval=interval,blit=True)
      # ani.save(PATHSAVE + pattern +'.mp4', dpi=600)  
      #writer.close()













