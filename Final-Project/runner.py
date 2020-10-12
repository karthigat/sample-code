

# Reference: https://github.com/guyuchao/Grabcut

import cv2
from PIL import Image
import numpy as np
from GMM import *
from math import log
from Kmeans import *
import copy
from gcgraph import *
from basicprocess import *
import base64

import os
import psutil
import xml.etree.ElementTree as ET
import sys

import matplotlib.pyplot as plt 

class grabcut(object):
    # print("step3")
    def __init__(self):
        # print("step5")
        self.cluster=5
        self.iter = 2
        self.BGD_GMM = None
        self.FGD_GMM = None
        self.KmeansBgd = None
        self.KmeansFgd = None
        self._gamma=50
        self._lambda=9*self._gamma
        self.GT_bgd=0#ground truth background
        self.P_fgd=1#ground truth foreground
        self.P_bgd=2#may be background
        self.GT_fgd=3#may be foreground

    #calculating  Beta for smootheness
    def Beta(self,npimg):
        # print("step6")
        rows,cols=npimg.shape[:2]

        ldiff = np.linalg.norm(npimg[:, 1:] - npimg[:, :-1])
        uldiff = np.linalg.norm(npimg[1:, 1:] - npimg[:-1, :-1])
        udiff = np.linalg.norm(npimg[1:, :] - npimg[:-1, :])
        urdiff = np.linalg.norm(npimg[1:, :-1] - npimg[:-1, 1:])
        beta=np.square(ldiff)+np.square(uldiff)+np.square(udiff)+np.square(urdiff)
        beta = 1 / (2 * beta / (4 * cols * rows - 3 * cols - 3 * rows + 2))
        # print(beta)
        return beta

    #estimating smoothness term
    def Smoothness(self, npimg, beta, gamma):
        # print("step7")
        rows,cols=npimg.shape[:2]
        self.lweight = np.zeros([rows, cols])
        self.ulweight = np.zeros([rows, cols])
        self.uweight = np.zeros([rows, cols])
        self.urweight = np.zeros([rows, cols])
        for y in range(rows):
            # print("stop1")
            for x in range(cols):
                color = npimg[y, x]
                if x >= 1:
                    diff = color - npimg[y, x-1]
                    # print(np.exp(-self.beta*(diff*diff).sum()))
                    self.lweight[y, x] = gamma*np.exp(-beta*(diff*diff).sum())
                if x >= 1 and y >= 1:
                    diff = color - npimg[y-1, x-1]
                    self.ulweight[y, x] = gamma/np.sqrt(2) * np.exp(-beta*(diff*diff).sum())
                if y >= 1:
                    diff = color - npimg[y-1, x]
                    self.uweight[y, x] = gamma*np.exp(-beta*(diff*diff).sum())
                if x+1 < cols and y >= 1:
                    diff = color - npimg[y-1, x+1]
                    self.urweight[y, x] = gamma/np.sqrt(2)*np.exp(-beta*(diff*diff).sum())

    #creating GMM for foreground and background
    def init_with_kmeans(self,npimg,mask):
        print("Creating GMM.....")
        # print("step8")
        self._beta = self.Beta(npimg)
        self.Smoothness(npimg, self._beta, self._gamma)

        bgd = np.where(mask==self.GT_bgd)
        prob_fgd = np.where(mask==self.P_fgd)
        BGDpixels = npimg[bgd]#(_,3)
        FGDpixels = npimg[prob_fgd]#(_,3)

        self.KmeansBgd = Kmeans(BGDpixels, dim=3, cluster=5, epoches=2)
        self.KmeansFgd = Kmeans(FGDpixels, dim=3, cluster=5, epoches=2)
        

        bgdlabel=self.KmeansBgd.run() # (BGDpixel.shape[0],1)
        # print(bgdlabel)
        fgdlabel=self.KmeansFgd.run() # (FGDpixel.shape[0],1)
        # print(fgdlabel)

        self.BGD_GMM = GMM()  # The GMM Model for BGD
        self.FGD_GMM = GMM()  # The GMM Model for FGD


        for idx,label in enumerate(bgdlabel):
            self.BGD_GMM.add_pixel(BGDpixels[idx],label)
        for idx, label in enumerate(fgdlabel):
            self.FGD_GMM.add_pixel(FGDpixels[idx], label)

        # learning GMM parameters
        self.BGD_GMM.learning()
        self.FGD_GMM.learning()

    # initial call
    def __call__(self,epoches,npimg,mask):
        print("Starting.....")
        # print("step9")
        self.init_with_kmeans(npimg,mask)
        for epoch in range(epoches):
            self.assign_step(npimg,mask)
            self.learn_step(npimg,mask)
            self.construct_gcgraph(npimg,mask)
            mask = self.estimate_segmentation(mask)
            img = copy.deepcopy(npimg)
            img[np.logical_or(mask == self.P_bgd, mask == self.GT_bgd)] = 0
        return Image.fromarray(img.astype(np.uint8))

    # assigning GMMs parameters
    def assign_step(self,npimg,mask):
        print("Assinging GMM parameter.....")
        # print("step10")
        rows,cols=npimg.shape[:2]
        clusterid=np.zeros((rows,cols))
        for row in range(rows):
            for col in range(cols):
                pixel=npimg[row,col]
                if mask[row,col]==self.GT_bgd or mask[row,col]==self.P_bgd:#bgd
                    clusterid[row,col]=self.BGD_GMM.pixel_from_cluster(pixel)
                else:
                    clusterid[row, col] = self.FGD_GMM.pixel_from_cluster(pixel)
        self.clusterid=clusterid.astype(np.int)

    #Learning GMM parameter
    def learn_step(self,npimg,mask):
        print("Learning parameter......")
        # print("step11")
        for cluster in range(self.cluster):
            bgd_cluster=np.where(np.logical_and(self.clusterid==cluster,np.logical_or(mask==self.GT_bgd,mask==self.P_bgd)))
            fgd_cluster=np.where(np.logical_and(self.clusterid==cluster,np.logical_or(mask==self.GT_fgd,mask==self.P_fgd)))
            for pixel in npimg[bgd_cluster]:
                self.BGD_GMM.add_pixel(pixel,cluster)
            for pixel in npimg[fgd_cluster]:
                self.FGD_GMM.add_pixel(pixel,cluster)
        self.BGD_GMM.learning()
        self.FGD_GMM.learning()

    # constructing graph
    def construct_gcgraph(self,npimg,mask):
        print("Graph construction...may take a while.....")
        # print("step12")
        rows,cols=npimg.shape[:2]
        vertex_count = rows*cols
        edge_count = 2 * (4 * vertex_count - 3 * (rows + cols) + 2)
        self.graph = GCGraph(vertex_count, edge_count)
        for row in range(rows):
            for col in range(cols):
                #source background sink foreground
                vertex_index = self.graph.add_vertex()
                color = npimg[row, col]
                if mask[row, col] == self.P_bgd or mask[row, col] == self.P_fgd:#pred fgd
                    fromSource = -log(self.BGD_GMM.pred_GMM(color))
                    toSink = -log(self.FGD_GMM.pred_GMM(color))
                elif mask[row, col] == self.GT_bgd:
                    fromSource = 0
                    toSink = self._lambda
                else:
                    fromSource = self._lambda
                    toSink = 0
                self.graph.add_term_weights(vertex_index, fromSource, toSink)

                if col-1 >= 0:
                    w = self.lweight[row, col]
                    self.graph.add_edges(vertex_index, vertex_index - 1, w, w)
                if row-1 >= 0 and col-1 >= 0:
                    w = self.ulweight[row, col]
                    self.graph.add_edges(vertex_index, vertex_index - cols - 1, w, w)
                if row-1 >= 0:
                    w = self.uweight[row, col]
                    self.graph.add_edges(vertex_index, vertex_index - cols, w, w)
                if col+1 < cols and row-1 >= 0:
                    w = self.urweight[row, col]
                    self.graph.add_edges(vertex_index, vertex_index - cols + 1, w, w)
    # segmentation estimation E( α , k, θ , z) - min cut
    def estimate_segmentation(self,mask):
        print("Estimation.......")
        # print("step13")
        rows,cols=mask.shape
        self.graph.max_flow()
        for row in range(rows):
            for col in range(cols):
                if mask[row, col] == self.P_fgd or mask[row,col]==self.P_bgd :
                    if self.graph.insource_segment(row * cols + col):  # Vertex Index
                        mask[row, col] = self.P_fgd
                    else:
                        mask[row, col] = self.P_bgd
        # print("working")

        # self.KmeansBgd.plot()
        # self.KmeansFgd.plot()
        
        return mask

if __name__=="__main__":

    call_grabcut=grabcut() 

    # getting annotation and image dataset
  
    def get_directory(a_dir):
        return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
    rootdir = os.getcwd()
    Annotation = "Annotation"
    images = []
    annotations = []
    for directory in get_directory(rootdir):
        if directory == 'Annotation' or directory == 'images':
            for thisroot, _, files in os.walk(os.path.join(rootdir, directory)):
                for filename in files:
                    if directory == 'Annotation':
                        annotations.append(os.path.join(thisroot, filename))
                    if directory == 'images':
                        images.append(os.path.join(thisroot, filename))
    images = sorted(images)
    annotations = sorted(annotations)

    dataset = []
    for i in range(len(images)):
        dataset.append((images[i], annotations[i]))
    
    # creating directory for saving segmented image

    for j in range(len(dataset)):

       image = images[j]
       annotation = annotations[j]

       basename = os.path.basename(image)
       filename = os.path.splitext(basename)

       path =  "result"
       if os.path.exists(os.path.join(rootdir,path)):
           pass
       else:
           make_dir = os.mkdir(path)
           print("Directory created....")

       dir_path = os.path.join(path,filename[0]+".png")

          
       img = np.array(Image.open(image)).astype(np.float32)
       annotation_img = open(annotation)
       root = ET.fromstring(annotation_img.read())
       annotation_img.close()

       xmin = int (root.find('object').find('bndbox').find('xmin').text)
       ymin = int (root.find('object').find('bndbox').find('ymin').text)
       xmax = int (root.find('object').find('bndbox').find('xmax').text)
       ymax = int (root.find('object').find('bndbox').find('ymax').text)
       height = int (root.find('size').find('height').text)
       width = int (root.find('size').find('width').text)
       depth = int (root.find('size').find('depth').text)

       mask = np.zeros((height, width) , dtype = int)

       x_min = xmin
       y_min = ymin
       x_max = xmax
       y_max = ymax

       mask[y_min:y_max, x_min:x_max] = call_grabcut.P_fgd
       

       ret = call_grabcut(epoches=2,npimg=img,mask=mask) # calling function _call_
       imge = Basicprocess().img2base64(ret)

       with open(dir_path, "wb") as fh:
           fh.write(base64.b64decode(imge))
    #    cv2.waitKey()
       print("Successfully segmented....")
       print("********************")
    
    #    graph = self.Kmeans.plot()
    #    print(graph)

    #    pid = os.getpid()
    #    process = psutil.Process(pid)
    #    memoryuse = process.memory_info()[0]/2.**30 
       
    #    print("Time taken for segmenting image".format(fh.mins))
    #    print('memory usage:', memoryuse)
    #    print("CPU percentage:", psutil.cpu_percent())

    # img=np.array(Image.open("pexels-photo-1108099.jpeg")).astype(np.float32)
    # # print(img[400,500,2]) 
    # print("step1")
    # gg=grabcut() 
    # # print(gg)
    # print(img.shape[:2])
    # mask = np.zeros(img.shape[:2])
    # # cv2.rectangle(img,(x, y), (x + w, y + z), (36,255,12), 2)
    # print(mask.shape)
    # # print(mask)
    # left = 34
    # right = 315
    # top = 66
    # bottom = 374
    # cv2.imshow("im",img1)
    # mask[left:right, top:bottom] = gg.P_fgd
    # print(mask[left:right, top:bottom].shape)
    # # cv2.imshow("im", mask)
    # print(gg.P_fgd)
    # print("step2")
    # ret = call_grabcut(epoches=1,npimg=img,mask=mask) # calling function _call_
    # imge = Basicprocess().img2base64(ret)
    # print(imge)
    # print("step4")
    # # cv2.imshow("im", imge)
    
    # with open("imageToSave.png", "wb") as fh:
    #     fh.write(base64.b64decode(imge))
    # cv2.waitKey()




