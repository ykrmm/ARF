"""Fichier des modèles de prediction pour le TME2. 


- Modèle des histogrammes
- Modèle des noyaux de Parzen 
- Modèle des noyaux Gaussiens. 



"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import operator
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



class histogramme:
    
    def __init__(self,data):
        self.data = data
        self.mat = []
    def fit(self,poi,X,Y,pas):
        self.pas = pas
        self.xmin = X[0]
        self.xmax = X[1]
        self.ymin = Y[0]
        self.ymax = Y[1]
        self.xinter = (X[1]-X[0])/pas
        self.yinter = (Y[1]-Y[0])/pas
        N = len(self.data[poi]) # Le nombre de point d'intéret poi en région parisienne
        self.mat = np.zeros((pas,pas)) # création d'une matrice de taille pas x pas qui va contenir nos estimations de densité 
    
        
        for clef in self.data[poi]:
            coord = self.data[poi][clef][0]
            
            y= int((coord[0]-Y[0])/self.yinter)
            x = int((coord[1]-X[0])/self.xinter)
            self.mat[y][x] += 1
        
        
       
        self.mat = np.divide(self.mat,(N*pas)) # on divise par N? N*pas ? N*pas**2 ? 
       
        return self.mat

        
            
    
    def predict(self,coord):
        if coord[0] < self.ymin or coord[1] < self.xmin or coord[0] > self.ymax or coord[1] > self.xmax : 
            raise ValueError('coords must be between xmin xmax and ymin ymax')
            
        else:
             y= int((coord[0]-self.ymin)/self.yinter)
             x = int((coord[1]-self.xmin)/self.xinter)
             density = self.mat[y][x]
             
             print( "The density at this point ",coord," is ",density)
             return density 
            
    def affichage_3D(self):
        """marche pas """
        print(self.ymin)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xpos = [self.xmin+ self.xinter for i in range(self.pas) ]
        ypos = [self.ymin+ self.yinter for i in range(self.pas) ]
        zpos = 0 
        print(self.mat)
        dx = np.ones_like(xpos)
        dy = np.ones_like(ypos)
        dz = self.mat.ravel()
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

       
        plt.show()
        plt.savefig('hist3D.png')
        
        
class noyau_parzen:
    
    
    def __init__(self,data,X,Y,poi):
        self.data = data
        self.xmin = X[0]
        self.xmax = X[1]
        self.ymin = Y[0]
        self.ymax = Y[1]
        self.poi = poi
        self.N = len(self.data[self.poi])
        
    def indicatrice_phi(self,coord):
        phi = np.sqrt(sum([i**2 for i in coord]))
        #print(phi)
        if phi <=0.5:
            return 1
        else:
            return 0
        
    
    def densite(self,h,yx): 
        """
        On donne en paramètre le point d'intéret
        ainsi que h la longueur de l'hypercube.
        
        Renvoie l'estimation de densité au point yx.
        """
        
        
        k = 0
        for clef in self.data[self.poi]:
            coord = self.data[self.poi][clef][0]
            d = tuple(np.divide(tuple(map(operator.sub, yx, coord)),h)) # Pour chaque coordonnées des pts d'intérets on calcule la difference d avec la coordonnée yx
            
            k+= (self.indicatrice_phi(d))/h**2
            
        
        return k/(self.N)
    
    
    def estimation_parzen(self,h):
        self.h = h #LONGUEUR HYPERCUBE. 
        nbcubex = int((self.xmax -self.xmin)/self.h)   # Nombre d'hypercube de longueur h qu'on peut mettre sur l'axe x. 
        nbcubey = int((self.ymax - self.ymin)/self.h)  # Nombre d'hypercube de longueur h qu'on peut mettre sur l'axe y.
      
        
        
        self.mat = np.zeros((nbcubey,nbcubex)) # Va contenir les estimations de densités centré sur les hypercubes
       
        
        
        for j in range(0,nbcubey):
            for i in range(0,nbcubex):
                x = self.xmin + i*self.h
                y = self.ymin + j*self.h
                point = (y+(self.h/2),x+(self.h/2)) # Le point est le centre de l'hypercube c'est ici qu'on calcul la densité
                d = self.densite(self.h,point)
                self.mat[j][i] =d
        return self.mat


    def affichage_3D(self):
        
        
        
       
        
        # Make data.
        X = np.arange(self.xmin, self.xmax, self.h)
        Y = np.arange(self.ymin, self.ymax, self.h)
       
        Z = np.zeros((len(Y),len(X)))
        
        X, Y = np.meshgrid(X, Y)
        
        for j in range(len(Y)):
            for i in range(len(X)):
                x = self.xmin + i*self.h
                y = self.ymin + j*self.h
                point = (y+(self.h/2),x+(self.h/2)) # Le point est le centre de l'hypercube c'est ici qu'on calcul la densité
                d = self.densite(self.h,point)
                Z[j][i] = d
                
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,cmap=cm.RdBu,linewidth=0, antialiased=False)
        
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        fig1 = plt.gcf()
        
        plt.show()
        fig1.savefig('parzen3Dh='+str(self.h)+'.png',dpi=100)
        plt.close()
               
        
        
            
            
            
            
            
            
            
            
            
            
            