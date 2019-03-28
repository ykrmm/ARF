from arftools import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def mse(datax,datay,w):
    """ retourne la moyenne de l'erreur aux moindres carres """
    resultat = 0
    taille = len(datax)
    for i in range(taille):
        resultat += (np.matmul(datax[i],w)-datay[i])**2
    return resultat/taille

def mse_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur au moindres carres """
    resultat = 0
    taille = len(datax)
    for i in range(taille):
        resultat += 2*np.matmul(datax[i],w)-2*sum(datay[i]*datax[i])
    return resultat/taille

def hinge(datax,datay,w):
    """ retourn la moyenne de l'erreur hinge """
    
    #return np.maximum(np.zeros((testy.shape)),np.dot(testy,np.dot(testx,w.T)))
    
    resultat = 0
    taille = len(datax)
    for i in range(taille):
        resultat+=(max(0,-datay[i]*np.matmul(datax[i],w.T)))
    return resultat/taille

def hinge_g(datax,datay,w):
    """ retourne le gradient moyen de l'erreur hinge """
    resultat = np.zeros((w.shape))
    taille = len(datax)
    for i in range(taille):
        if -datay[i]*np.dot(datax[i],w.T) < 0:
            resultat += -datay[i]*datax[i]
    return resultat/taille

class Lineaire(object):
    
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g

    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        # on transforme datay en vecteur colonne
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))
        for i in range(self.max_iter):
            self.w = self.w - self.eps*self.loss_g(datax,datay,self.w)
           
        

    def predict(self,datax):
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        return np.sign(np.matmul(datax,self.w.T))

    def score(self,datax,datay):
        score = 0
        taille = len(datax)
        for i in range(taille):
            if self.predict(datax[i]) == datay[i]:
                score += 1 
        return score/taille

        



def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")



def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()



if __name__=="__main__":
    """ Tracer des isocourbes de l'erreur """
    plt.ion()
    trainx,trainy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    testx,testy =  gen_arti(nbex=1000,data_type=0,epsilon=1)
    plt.figure()
    plot_error(trainx,trainy,mse)
    plt.figure()
    plot_error(trainx,trainy,hinge)

    #testx = np.hstack((testx,np.ones((testx.shape[0],1)))) # On ajoute un biais
    #trainx = np.hstack((trainx,np.ones((trainx.shape[0],1)))) # On ajoute un biais

    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    perceptron.fit(trainx,trainy)
    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)

 