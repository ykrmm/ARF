import numpy as np





class Loss(object):
    
    def forward(self,y,yhat):
        #Calcul du cout 
        return (yhat -y)**2
    
    def backward_update_gradient(self,y,yhat):
        #Calcul le gradient du cout
        return 2*(yhat-y)
    
    
class Module(object):
    
    def __init__(self):
        self._parameters = None # Poids
        self._gradient = None
        
    def zero_grad(self):
        pass
        
    def forward(self,X):
        ## Calcul la passe forward
        pass        
    def update_parameters(self,gradient_step):
        #Calcul la maj des parametres selon le grad calculé et le pas de gradient_step
        pass
    def backward_update_gradient(self,entree,delta):
        #Met à jour la valeur du gradient
        pass        
        
    def backward_delta(self,entree,delta):
        #Calcul la derivee de l'erreur
        pass

        
class Lineaire(Module):
    def __init__(self):
        self._parameters = None # Poids
        self._gradient = None
        
    def zero_grad(self):    
        self._gradient = 0
        
    def forward(self,X):
        ## Calcul la passe forward     
        return np.matmul(X,self._parameters)
    

class Tanh(Module):

            
    def forward(self,X):
        ## Calcul la passe forward     
        return np.tanh(X)
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":

    pass    