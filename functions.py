import numpy as np
import requests
import random

def extract_data(mon_fichier):
    with open(mon_fichier, "r") as fichier:
        contenu = fichier.read()
        contenu = contenu.split()
        #print(contenu)
        nb_instances = int(contenu[0])
        #print(nb_instances)
        instances = {}
        compteur = 1
        for i in range(1,nb_instances+1):
            elements = contenu[compteur:]
            n  = int(elements[0])
            #print(n)
            m = int(elements[1])
            #print(m)
            val_opt = float(elements[2])
            cost = np.array([float(elements[e]) for e in range(3,3+n)])
            #print(cost)
            a = np.array([int(elements[e]) for e in range(3+n,3+n+n*m)])
            a = a.reshape(m,n)
            b = np.array([int(elements[e]) for e in range(3+n+n*m,3+n+n*m+m)])
            instances[i] = {}
            instances[i]["Nb projet"] = n
            instances[i]["Nb ressource"] = m
            instances[i]["Valeur optimale"] = val_opt
            instances[i]["cost"] = cost
            instances[i]["A"] = a
            instances[i]["B"] = b
            compteur += 3+n+m*n+m
        return nb_instances, instances


def heuristique_sac_a_dos(n, m, cost, a, b, fct_voisinage):
    x= np.zeros(n)
    #print('xtype', type(x)) 
    b_prime = np.sum(b) #somme des ressources
    #print('b_prime=', b_prime) 
    a_prime= np.sum(a, axis=0) #somme des ressources nécessaires pour chaque projet
    #print('a_prime=', a_prime)
    y = -cost/a_prime #signe - pour trier dans l'ordre décroissant
    #print(y)
    indices =y.argsort()
    #print('indices', indices)
    ressource = 0
    index = []
    for i in indices:
        if ressource + a_prime[i] <= b_prime:
            x[i] = 1
            ressource += a_prime[i]
            #print('i=', i)
            #print('ressource', ressource)
            #print('b_prime', b_prime)
            index.append(i) 
    #print("x before dot (a,x) ", x)
    #verification solution réalisable + réparation
    while any(np.dot(a, x) > b):
        #print('x=', x)
        #print('p=',(np.dot(a, x)))
        #print('b=',b)
        last_index = index[-1]
        x[last_index] = 0
        #print(x)
        index.pop()
    #print("x after dot (a,x) ", x)
    value = np.dot(cost, x)

    return x, value


def is_realisable(x, a, b):
    return all(np.dot(a, x) <= b)


def voisinage(x):
    voisins = []
    for i in range(len(x)):
        if x[i] == 0:
            x_voisin = x.copy()
            x_voisin[i] = 1
            voisins.append(x_voisin)
        else:
            x_voisin = x.copy()
            x_voisin[i] = 0
            voisins.append(x_voisin)
    return voisins


def voisinage_echange(x):
    voisins = []
    for i in range(len(x)):
        if x[i] == 1:
            for j in range(len(x)):
                if x[j] == 0:
                    x_voisin = x.copy()
                    x_voisin[i] = 0
                    x_voisin[j] = 1
                    voisins.append(x_voisin)
    return voisins


def voisin_realisable(x, a, b, fct_voisinage):
    voisins = fct_voisinage(x)
    voisins_realisables = []
    for voisin in voisins:
        if is_realisable(voisin, a, b):
            voisins_realisables.append(voisin)
    return voisins_realisables


def algorithme_montee(xmax, a, b, cost, fct_voisinage):
    fin = False
    while not fin:
        voisins = voisin_realisable(xmax, a, b, fct_voisinage)
        fin = True
        value = np.dot(cost, xmax)
        for voisin in voisins:
            value_voisin = np.dot(cost, voisin)
            if value_voisin > value:
                xmax = voisin
                value = value_voisin
                fin = False
    return xmax, value


def perturber_solution(x):
    n = x.size
    for _ in range(np.random.randint(1, 2+ n//10)):
        index = np.random.randint(len(x))
        x[index] = 1 - x[index]
    return x


def fit(x, a, b, cost):
    if not is_realisable(x, a, b):
        return -1
    return np.dot(cost, x)


def reparation_v2(x, a, b, cost):
    b_prime = np.sum(b) 
    a_prime= np.sum(a, axis=0) 
    y = -cost/a_prime 
    indices =y.argsort()
    index_to_pop = [i for i in indices if x[i] == 1]
    while any(np.dot(a, x) > b):
        last_index = index_to_pop[-1]
        x[last_index] = 0
        #print(x)
        index_to_pop.pop()
    return x

def generation_pop(n, m, cost, a, b, taille_pop, generation_solution, fct_voisinage):   
    x = generation_solution(n, m, cost, a, b, fct_voisinage)[0]
    #x = [int(i) for i in x]
    popu = [x]
    for _ in range (taille_pop-1):
        x = perturber_solution(x)
        popu.append(x)
    return popu


def algorithme_genetique(n, m, cost, a, b, nb_iter, taille_pop, max_pop, taux_mut, generation_solution, fct_voisinage):
    max_pop = (max_pop//2)*2
    #population initiale
    pop = generation_pop(n, m, cost, a, b, taille_pop, generation_solution, fct_voisinage)
    random.shuffle(pop)
    #iterations
    for _ in range(nb_iter):
        fit_value = [fit(x, a, b, cost) for x in pop]
        proba = [fit_value[i]/sum(fit_value) for i in range(len(fit_value))]
        parents = []
        #selection parents
        while len(parents) < max_pop:
            for i in range(len(pop)):
                if np.random.rand() < proba[i]:
                    parents.append(pop[i])
        parents = parents[:max_pop]
        #croisement
        #1-point crossover
        child = []
        for i in range(0, len(parents), 2):
            cut = np.random.randint(1, n)
            #print(len(parents[i]))
            #print(len(parents[i+1]))
            child1 = np.concatenate((parents[i][:cut], parents[i+1][cut:]))
            child2 = np.concatenate((parents[i+1][:cut], parents[i][cut:]))
            #print('x',child2.shape)
            child.append(child1)
            child.append(child2)
        #2-point crossover
        random.shuffle(parents)
        for i in range(0, len(parents), 2):
            p1, p2 = sorted(np.random.choice(range(n), size=2, replace=False))
            child1 = np.concatenate((parents[i][:p1], parents[i+1][p1:p2], parents[i][p2:]))
            child2 = np.concatenate((parents[i+1][:p1], parents[i][p1:p2], parents[i+1][p2:]))
            child.append(child1)
            child.append(child2)
            #print('x',child2.shape)
        for j in range(len(child)):
            if not is_realisable(child[j], a, b):
                child[j] = reparation_v2(child[j], a, b, cost)
        #mutation
        for j in range(len(child)):
            if np.random.rand() < taux_mut:
                child[j] = perturber_solution(child[j])
        pop = parents + child
        random.shuffle(pop)
    valeurs = [fit(x, a, b, cost) for x in pop]
    index = valeurs.index(max(valeurs))
    x= pop[index]
    value = np.dot(cost, pop[index])
    return x, value
