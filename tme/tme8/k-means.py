#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:52:33 2019

@author: 3775070
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance
class Kmeans():
    
    def __init__(self, nombre_cluster,nb_it,eps):
        """
            eps : petit nombre critère de convergence 
            nb_it : nombre d'itérations max
            
        """
        self.nb_cluster = nombre_cluster
        self.eps = eps
        self.nb_it = nb_it
    def calcul_baricentre(self,data):
        for i in range (self.nb_cluster):
            points_cluster = data[self.which_cluster == i]
            if len(points_cluster)!=0:
                self.cluster[i]=np.mean(points_cluster,axis=0) 

    def fit(self, data):

        self.cluster =np.array([np.random.randint(0, 256, 3) for i in range(self.nb_cluster)] ) # array de cluster
        self.which_cluster = np.zeros((len(data),))
        for i in range(self.nb_it):
            d = scipy.spatial.distance.cdist(data,self.cluster,metric='euclidean')
            ind_clust = np.argmin(d,axis=1)
            self.which_cluster = ind_clust
           
            tmp = self.cluster.copy()
            self.calcul_baricentre(data)
            
            if np.absolute(np.sum(tmp-self.cluster)) < self.eps:
                break
            
    def predict(self,data):
        """
            À voir s'il y a besoin.
        """
        which_cluster = []
        
        
        d = scipy.spatial.distance.cdist(data,self.cluster,metric='euclidean')
        ind_clust = np.argmin(d,axis=1)
        for i in ind_clust:
            which_cluster.append(self.cluster[i])
        return np.array(which_cluster)
            
        


if __name__ == '__main__':
    
    im = plt.imread("beau_gosse.jpeg")[:,:,:3]  # on garde que les 3 premieres composantes, la transparence est inutile
    im_h, im_l ,_ = im.shape
    
    pixels = im.reshape((im_h*im_l,3))  #transformation  en  matrice n∗3, n nombre de pixels
    imnew = pixels.reshape((im_h, im_l , 3 ) ) #transformation  inverse
    plt.imshow(im) #affihcer l’image
    
    k_means = Kmeans(256,200,1)
    k_means.fit(pixels)
    image = k_means.predict(pixels)
    image = image.reshape((im_h,im_l,3))
    plt.imshow(image)
