"""Fichier des modèles de prediction pour le TME2. 


- Modèle des histogrammes
- Modèle des noyaux de Parzen 
- Modèle k plus proches voisins k-NN. 



"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt





class histogramme:
    
    def __init__(self,data):
        self.data = data
    
    def fit(self,poi,X,Y,pas):
        xinter = (X[1]-X[0])/pas
        yinter = (Y[1]-Y[0])/pas
        N = len(self.data[poi]) # Le nombre de point d'intéret poi en région parisienne
        mat = np.zeros((pas,pas)) # création d'une matrice de taille pas x pas qui va contenir nos estimations de densité 
    
        
        for clef in self.data[poi]:
            coord = self.data[poi][clef][0]
            
            y= int((coord[0]-Y[0])/yinter)
            x = int((coord[1]-X[0])/xinter)
            mat[y][x] += 1
        
        
       
        mat = np.divide(mat,(N*pas)) # on divise par N? N*pas ? N*pas**2 ? 
        """ fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xpos = [X[0]+ xinter for i in range(pas) ]
        ypos = [Y[0]+ yinter for i in range(pas) ]
        zpos = 0 
        plt.hist2d(mat)
        plt.show()"""
        return mat

        
            
    
    def predict(self,grid):
        pass            