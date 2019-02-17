
import  pickle
import  numpy as np
import random
import matplotlib.pyplot as plt
# data : tableau (films ,features), id2titles : dictionnaire  id -> titre ,
# fields : id  feature  -> nom
[data , id2titles , fields ]= pickle.load(open("imdb_extrait.pkl","rb"))
# la  derniere  colonne  est le vote
datax=data [: ,:32]
datay=np.array ([1 if x[33] >6.5  else  -1 for x in data])



from  decisiontree  import  DecisionTree
dt = DecisionTree ()
dt.max_depth = 5
#on fixe la  taille  de l’arbre a 5
dt.min_samples_split = 2
#nombre  minimum d’exemples  pour  spliter  un noeud
dt.fit(datax ,datay)
dt.predict(datax [:5 ,:])
print(dt.score(datax ,datay))
# dessine l’arbre  dans un  fichier  pdf   si pydot  est  installe.
dt.to_pdf("/Users/ykarmim/Documents/Cours/Master/M1S2/ARF/tme/tme1/test_tree.pdf",fields)
# sinon  utiliser  http :// www.webgraphviz.com/
dt.to_dot(fields)
#ou dans la  console
print(dt.print_tree(fields ))

def partitionnement_test(datax,datay,rp,rdm): #rp la proportion qui sera dans le test.
    
    dt = DecisionTree()
    dt.min_samples_split = 2
    if rdm:
        rp = random.uniform(0,1)
    indiceap = np.random.choice(np.arange(len(datax)), int(rp*len(datax)), replace = False)
    indicet = []
    for i in range(0,len(datax)):
        if i not in indiceap:
            indicet.append(i)
    testy = np.zeros((len(indicet)), int)
    apprentissagey = np.zeros((len(indiceap)),int)
    
       
    
    
    testx = np.delete(datax,indiceap,axis=0)
    
    apprentissagex = np.delete(datax,indicet,axis=0)
    
    for i in range(0,len(indiceap)):
        apprentissagey[i] = datay[indiceap[i]]
    for i in range(0,len(indicet)):
        testy[i] = datay[indicet[i]]
    
    
    l_scoretest = []
    l_scoreapprentissage = []
    
    for i in range(2,20,3):
        dt.max_depth = i
        dt.fit(apprentissagex ,apprentissagey)
        dt.predict(apprentissagex[:5 ,:])
        l_scoretest.append(1 - dt.score(testx,testy))
        l_scoreapprentissage.append(1 - dt.score(apprentissagex,apprentissagey))
    plt.plot(range(2,20,3),l_scoretest,'r--',range(2,20,3),l_scoreapprentissage,'b--')
    plt.show()
    plt.close()