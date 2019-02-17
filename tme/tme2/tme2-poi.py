import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

class histogramme:
    
    def __init__(self,data):
        self.data = data
    
    def fit(self,poi,X,Y,pas=100):
        xinter = (X[1]-X[0])/pas
        yinter = (Y[1]-Y[0])/pas
        
        mat = np.zeros((pas,pas))
    
        
        for clef in self.data[poi]:
            coord = self.data[poi][clef][0]
            mat(,)
            y[int((coord[0]-Y[0])/yinter)]+=1
            x[int((coord[1]-X[0])/xinter)]+=1
        
        
        N = len(self.data[poi])
        x = np.true_divide(x,(N*pas**2))
        y = np.true_divide(y,(N*pas**2))
        
        plt.hist2d(x, y)
        plt.show()
        return 

        
            
    
    def predict(self,grid):
        pass            
        
    
    
    
    
    
plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    ## extent pour controler l'echelle du plan

poidata = pickle.load(open("data/poi-paris.pkl","rb"))
## liste des types de point of interest (poi)
print("Liste des types de POI" , ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "restaurant"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]),2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]

## Affichage brut des poi
show_map()
## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)


###################################################

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 100
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
grid = np.c_[xx.ravel(),yy.ravel()]

# A remplacer par res = monModele.predict(grid).reshape(steps,steps)
res = histogramme(poidata).fit("restaurant",[2.23,2.48],[48.806,48.916]).reshape(steps,steps)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.scatter(geo_mat[:,0],geo_mat[:,1],extent=[xmin,xmax,ymin,ymax],alpha=0.3)


