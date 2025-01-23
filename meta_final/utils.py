import numpy as np
import requests
import random
import matplotlib.pyplot as plt
import random
from voisinage import is_realisable, voisin_realisable
from reparation import reparation

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

def selection_pop(pop, values):
    proba = [values[i]/sum(values) for i in range(len(values))]
    a=0
    while a==0:
        for i in range(len(pop)):
            if np.random.rand() < proba[i]:
                x = pop[i]
                a=1
    return x


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
        index = index[:-1]
    #print("x after dot (a,x) ", x)
    value = np.dot(cost, x)

    return x, value


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

## pertube la solution 
def perturber_solution(x, proba_pertubation):
    n = x.size
    for _ in range(n):
        if np.random.rand() < proba_pertubation:
            index = np.random.randint(n)
            x[index] = 1 - x[index]
    return x


def fit(x, a, b, cost):
    method_fit = '0'
    # Vérifier si la solution est réalisable
    if method_fit=='1':  ## méthode 1 : pénalisation en fonction de la ressource utilisée
        if not is_realisable(x, a, b):
                # Pénaliser les solutions non réalisables
            penalite = np.sum(np.maximum(0, np.dot(a, x) - b))
            penalite_norm = penalite / (np.sum(b) + 1e-6)  # Normalisation
            penalite = penalite_norm * np.dot(cost, x)  # Pondération
            valeur = np.dot(cost, x)
            return max(valeur - penalite, 0)

        # Si la solution est réalisable, calculer la récompense
        valeur = np.dot(cost, x)
        used_ressources = np.sum(np.dot(a, x))
        free_ressources = np.sum(b)
        
        # Récompense basée sur la minimisation des ressources utilisées
        reward = free_ressources - used_ressources
        reward_norm = reward / (np.sum(b) + 1e-6)  # Normalisation
        reward = min(reward_norm * valeur, 0.03 * valeur)  # Limiter la récompense à 3% de la valeur
        return valeur + reward
    
    else: ## méthode 2 : pénalisation par 0 si non réalisable
        if not is_realisable(x, a, b):
            return 0
        return np.dot(cost, x)

def generation_pop(n, m, cost, a, b, taille_pop, generation_solution, fct_voisinage, sol_init, proba_pertubation): 
    if sol_init == 'random':
        sol_init = np.random.randint(0, 2, n)
        if not is_realisable(sol_init, a, b):
            sol_init = reparation(sol_init, a, b, cost)
    else :
        sol_init = generation_solution(n, m, cost, a, b, fct_voisinage)[0]
        sol_init = algorithme_montee(sol_init, a, b, cost, fct_voisinage)[0]

    #y = x.copy()
    popu = [sol_init]
    
    generation_init = '0' ## 2 méthodes de générationde la population initiale : de manière aléatoire ou en perturbant la solution initiale
    if generation_init == 'random':
        for _ in range (taille_pop-1):
            x = np.random.randint(0, 2, n)
            popu.append(x)

    else :
        for _ in range (taille_pop-1):
            x = sol_init.copy()
            x = mutate_solution(n, m, a, b, cost, x)
            popu.append(x)
    #print(np.array_equal(popu[0], popu[taille_pop-1]))
    
    return popu

## croisement utilisé dans la deuxième et troisième version de l'algorithme génétique
def croisement(parent1, parent2, cost, iteration):
    if iteration % 2 == 0:  # Croisement uniforme
        child1 = np.array([parent1[k] if np.random.rand() < 0.5 else parent2[k] for k in range(len(parent1))])
        child2 = np.array([parent2[k] if np.random.rand() < 0.5 else parent1[k] for k in range(len(parent1))])
    else:  # Croisement multi-points, avec sélection des élites
        contribution_parent1 = np.cumsum(parent1 * cost)
        contribution_parent2 = np.cumsum(parent2 * cost)

        start = min(np.argmax(contribution_parent1), len(parent1) - 1)
        stop = min(np.argmax(contribution_parent2), len(parent1))

        #print(np.array_equal(parent1, parent2))

        if stop < start:
            start, stop = stop, start
        elite_segment = slice(start, stop)

        child1 = np.concatenate((parent1[:elite_segment.start], parent2[elite_segment], parent1[elite_segment.stop:]))
        child2 = np.concatenate((parent2[:elite_segment.start], parent1[elite_segment], parent2[elite_segment.stop:]))
    return child1, child2

def mutate_solution(N, M, a, b, c, x):
    
    x_mut = x.copy()
    i = random.randint(0, len(x)-1)
    x_mut[i] = 1 - x_mut[i]
    x_mut = reparation(x_mut, a, b, c)

    return x_mut


### PREMIERE VERSION DE L'ALGORITHME GENETIQUE

def genetic_algo(n, m, cost, a, b, nb_iter, taille_pop, max_pop, taux_mut, generation_solution, fct_voisinage, sol_init, proba_pertubation):
    """ Méthode de recherche locale pour le problème du sac à dos multidimensionnel.

    Paramètres :
    c : Liste des gains associés aux projets.
    a : Matrice (M x N) des consommations de ressources.
    b : Liste des quantités disponibles de chaque ressource.
    max_iter : Nombre maximal d'itérations.
    pop_size : Taille de la population.

    Retourne :
    x_best : Meilleure solution trouvée.
    best_value : Gain total associé à la meilleure solution.
    """
    # ajouter critère d'arrêt
    #time = 0
    #population = gen_feasible_sols(n, m, a, b, cost, taille_pop)
    population = generation_pop(n, m, cost, a, b, taille_pop, generation_solution, fct_voisinage, sol_init, proba_pertubation)
    best_value = 0
    x_best = population[0]

    stats = {'iteration': [], 'max_fitness': [], 'mean_fitness': [], 'fitness_variance': []}

    for iter in range(nb_iter):

        #values = [sum(cost[j] * x[j] for j in range(n)) for x in population]
        values=[fit(x,a,b, cost) for x in population]

        best_idx = values.index(max(values))
        if values[best_idx] > best_value:
            x_best = population[best_idx]
            best_value = values[best_idx]

        max_fitness = max(values)
        mean_fitness = np.mean(values)
        fitness_variance = np.var(values)
        stats['iteration'].append(iter)
        stats['max_fitness'].append(max_fitness)
        stats['mean_fitness'].append(mean_fitness)
        stats['fitness_variance'].append(fitness_variance)


        parent1 = selection_pop(population, values)
        parent2 = selection_pop(population, values)
        child1 = croisement(parent1, parent2, cost,iter)[0]
        #print('child1',is_realisable(child1, a, b))
        #print('child2', is_realisable(child1, a, b))
        child2 = croisement(parent1, parent2, cost,iter)[1]
        x_mut1 = mutate_solution(n, m, a, b, cost, child1)
        x_mut2 = mutate_solution(n, m, a, b, cost, child2)

        array = np.array(values)
        index = np.argsort(array)[:4]

        population[index[0]] = x_mut1
        population[index[1]] = x_mut2
        population[index[2]] = child1
        population[index[3]] = child2

        #worst_idx = values.index(min(values))
        #population[worst_idx] = x_mut


    plt.figure(figsize=(10, 6))
    plt.plot(stats['iteration'], stats['max_fitness'], label='Max Fitness', linewidth=2)
    plt.plot(stats['iteration'], stats['mean_fitness'], label='Mean Fitness', linewidth=2)
    #plt.plot(stats['iteration'], stats['best_realisable'], label='Best Realisable Fitness', linewidth=2, linestyle='--')
    plt.fill_between(stats['iteration'], 
                     np.array(stats['mean_fitness']) - np.sqrt(stats['fitness_variance']), 
                     np.array(stats['mean_fitness']) + np.sqrt(stats['fitness_variance']), 
                     alpha=0.2, label='Variance Range')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Evolution of Population Statistics')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(is_realisable(x_best, a, b))
    if not is_realisable(x_best, a, b):
        x_best = reparation(x_best, a, b, cost)
    best_value = np.dot(x_best, cost)
    return x_best, best_value