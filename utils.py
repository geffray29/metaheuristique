import numpy as np
import requests
import random
import matplotlib.pyplot as plt
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
        index = index[:-1]
    #print("x after dot (a,x) ", x)
    value = np.dot(cost, x)

    return x, value


def is_realisable(x, a, b):
    return all(np.dot(a, x) <= b)

## 1 bit flip
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

## ce voisinage permutte deux bits de x
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

## renvoie un voisinage réalisable
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

## pertube la solution 
def perturber_solution(x, proba_pertubation):
    n = x.size
    for _ in range(n):
        if np.random.rand() < proba_pertubation:
            index = np.random.randint(n)
            x[index] = 1 - x[index]
    return x


def fit(x, a, b, cost):
    method_fit = '1'
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
        reward = min(reward_norm * valeur, 0.1 * valeur)  # Limiter la récompense à 10% de la valeur
        return valeur + reward
    
    else: ## méthode 2 : pénalisation par 0 si non réalisable
        if not is_realisable(x, a, b):
            return 0
        return np.dot(cost, x)


## Réparer les solutions non réalisables
def reparation(x, a, b, cost):
    b_prime = np.sum(b) 
    a_prime= np.sum(a, axis=0) 
    y = -cost/a_prime 
    indices =y.argsort() #trier les projets par ordre décroissant de coût par ressource
    index_to_pop = [i for i in indices if x[i] == 1]
    while any(np.dot(a, x) > b) and index_to_pop:
        last_index = index_to_pop[-1]
        x[last_index] = 0
        index_to_pop = index_to_pop[:-1]
    return x

def generation_pop(n, m, cost, a, b, taille_pop, generation_solution, fct_voisinage, sol_init, proba_pertubation): 
    if sol_init == 'random':
        x = np.random.randint(0, 2, n)
        if not is_realisable(x, a, b):
            x = reparation(x, a, b, cost)
    else :
        x = generation_solution(n, m, cost, a, b, fct_voisinage)[0]
        x = algorithme_montee(x, a, b, cost, fct_voisinage)[0]

    popu = [x.copy()]
    
    generation_init = '0' ## 2 méthodes de générationde la population initiale : de manière aléatoire ou en perturbant la solution initiale
    if generation_init == 'random':
        for _ in range (taille_pop-1):
            x = np.random.randint(0, 2, n)
            popu.append(x)

    else :
        for _ in range (taille_pop-1):
            x = perturber_solution(x, proba_pertubation)
            popu.append(x)
    print(np.array_equal(popu[0], popu[taille_pop-1]))
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

        print(np.array_equal(parent1, parent2))

        if stop < start:
            start, stop = stop, start
        elite_segment = slice(start, stop)

        child1 = np.concatenate((parent1[:elite_segment.start], parent2[elite_segment], parent1[elite_segment.stop:]))
        child2 = np.concatenate((parent2[:elite_segment.start], parent1[elite_segment], parent2[elite_segment.stop:]))
    return child1, child2


def mutation(solution, proba_pertubation):
    return np.array([
        1 - bit if np.random.rand() < proba_pertubation else bit
        for bit in solution
    ])


### PREMIERE VERSION DE L'ALGORITHME GENETIQUE

def algorithme_genetique(n, m, cost, a, b, nb_iter, taille_pop, max_pop, taux_mut, generation_solution, fct_voisinage, sol_init, proba_pertubation):
    max_pop = (max_pop//2)*2
    pop = generation_pop(n, m, cost, a, b, taille_pop, generation_solution, fct_voisinage, sol_init, proba_pertubation)
    random.shuffle(pop)
  
    for i in range(nb_iter):
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
            child1 = np.concatenate((parents[i][:cut], parents[i+1][cut:]))
            child2 = np.concatenate((parents[i+1][:cut], parents[i][cut:]))
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
        for j in range(len(child)):
            if np.random.rand() < taux_mut:
                child[j] = perturber_solution(child[j], proba_pertubation)
        pop = parents + child
        random.shuffle(pop)

    #sélection finale
    valeurs = [fit(x, a, b, cost) for x in pop]
    index_sorted = sorted(range(len(valeurs)), key = lambda i:valeurs[i])

    ##prendre une solution realisable
    method_real='1'
    if method_real=='1':
        pop[index_sorted[0]] = reparation(pop[index_sorted[0]], a, b, cost)  ##éviter la boucle infinie
        while not is_realisable(pop[index_sorted[-1]], a, b): ##prendre la meilleure solution réalisable
            index_sorted = index_sorted[:-1]
        x = pop[index_sorted[-1]]

    else:
        if not is_realisable(pop[index_sorted[-1]], a, b):
            pop[index_sorted[-1]] = reparation(pop[index_sorted[-1]], a, b, cost) ##réparer la meilleure solution
            x= pop[index_sorted[-1]]
        else:
            x = pop[index_sorted[-1]]
    value = np.dot(cost, x)
    return x, value


### DEUXIEME VERSION DE L'ALGORITHME GENETIQUE

def algorithme_genetique_withstats(
    n, m, cost, a, b, nb_iter, taille_pop, max_pop, taux_mut, 
    generation_solution, fct_voisinage, sol_init, proba_pertubation
):
    'Utilisation de nouveaux croisements et mutation adaptative'
    'Suivi des statistiques'    

    max_pop = (max_pop // 2) * 2
    # Population initiale
    pop = generation_pop(n, m, cost, a, b, taille_pop, generation_solution, fct_voisinage, sol_init, proba_pertubation)
    random.shuffle(pop)

    # Stockage des statistiques
    stats = {'iteration': [], 'max_fitness': [], 'mean_fitness': [], 'fitness_variance': [], 'best_realisable': []}

    # Itérations
    for iteration in range(nb_iter):
        # Calcul des fitness
        fit_value = [fit(x, a, b, cost) for x in pop]
        
        # Suivi des statistiques
        max_fitness = max(fit_value)
        mean_fitness = np.mean(fit_value)
        fitness_variance = np.var(fit_value)
        stats['iteration'].append(iteration)
        stats['max_fitness'].append(max_fitness)
        stats['mean_fitness'].append(mean_fitness)
        stats['fitness_variance'].append(fitness_variance)

        # Meilleure solution réalisable
        best_realisable = None
        for i in sorted(range(len(fit_value)), key=lambda k: fit_value[k], reverse=True):
            if is_realisable(pop[i], a, b):
                best_realisable = np.dot(cost, pop[i])
                break
        stats['best_realisable'].append(best_realisable if best_realisable is not None else 0)
        
        # Probabilités pour la sélection
        proba = [fit_value[i] / sum(fit_value) for i in range(len(fit_value))]
        parents = []

        # Sélection des parents
        while len(parents) < max_pop:
            for i in range(len(pop)):
                if np.random.rand() < proba[i]:
                    parents.append(pop[i])
        parents = parents[:max_pop]

        # Croisement et génération des enfants
        child = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = croisement(parent1, parent2, cost, iteration)
            child.append(child1)
            child.append(child2)

        # Mutation adaptative
        if fitness_variance < 1e-1:  # Détecter stagnation avec un seuil faible
            taux_mut_adaptatif = min(0.5, taux_mut * 2)  
        else:
            taux_mut_adaptatif = taux_mut

        for j in range(0,len(child),2):
            if np.random.rand() < taux_mut_adaptatif:
                child[j] = mutation(child[j], proba_pertubation)

        # Conserver les solutions élites 
        elite_count = max(1, len(pop) // 10)  # 10% des meilleurs individus
        elites_pop = sorted(pop, key=lambda x: fit(x, a, b, cost), reverse=True)[:elite_count]
        pop = elites_pop + parents + child
        pop = pop[:taille_pop]  # Réduire à la taille de population initiale
        random.shuffle(pop)


    # Sélection finale
    valeurs = [fit(x, a, b, cost) for x in pop]
    index_sorted = sorted(range(len(valeurs)), key = lambda i:valeurs[i])

    ##prendre une sol realisable
    method_real='1'
    if method_real=='1':
        pop[index_sorted[0]] = reparation(pop[index_sorted[0]], a, b, cost)  ##éviter la boucle infinie
        while not is_realisable(pop[index_sorted[-1]], a, b): ##prendre la meilleure solution réalisable
            index_sorted = index_sorted[:-1]
        x = pop[index_sorted[-1]]

    else:
        if not is_realisable(pop[index_sorted[-1]], a, b):
            pop[index_sorted[-1]] = reparation(pop[index_sorted[-1]], a, b, cost) ##réparer la meilleure solution
            x= pop[index_sorted[-1]]
        else:
            x = pop[index_sorted[-1]]
    value = np.dot(cost, x)
   
    # Génération du graphe des statistiques
    plt.figure(figsize=(10, 6))
    plt.plot(stats['iteration'], stats['max_fitness'], label='Max Fitness', linewidth=2)
    plt.plot(stats['iteration'], stats['mean_fitness'], label='Mean Fitness', linewidth=2)
    plt.plot(stats['iteration'], stats['best_realisable'], label='Best Realisable Fitness', linewidth=2, linestyle='--')
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

    return x, value




def algorithme_genetique_v3(
    n, m, cost, a, b, nb_iter, taille_pop, max_pop, taux_mut, 
    generation_solution, fct_voisinage, sol_init, proba_pertubation
):
    'Faire croiser les 2 meilleurs individus puis mutation'
    max_pop = (max_pop // 2) * 2
    # Population initiale
    pop = generation_pop(n, m, cost, a, b, taille_pop, generation_solution, fct_voisinage, sol_init, proba_pertubation)
    #random.shuffle(pop)
    x_best = pop[0]
    best_value=fit(pop[0],a,b,cost)
    # Stockage des statistiques
    stats = {'iteration': [], 'max_fitness': [], 'mean_fitness': [], 'fitness_variance': [], 'best_realisable': []}

    # Itérations
    for iteration in range(nb_iter):
        # Calcul des fitness
        values = [fit(x, a, b, cost) for x in pop]
        
        # Suivi des statistiques
        max_fitness = max(values)
        mean_fitness = np.mean(values)
        fitness_variance = np.var(values)
        stats['iteration'].append(iteration)
        stats['max_fitness'].append(max_fitness)
        stats['mean_fitness'].append(mean_fitness)
        stats['fitness_variance'].append(fitness_variance)

        # Meilleure solution réalisable
        best_realisable = None
        for i in sorted(range(len(values)), key=lambda k: values[k], reverse=True):
            if is_realisable(pop[i], a, b):
                best_realisable = np.dot(cost, pop[i])
                break
        stats['best_realisable'].append(best_realisable if best_realisable is not None else 0)

        best_idx = values.index(max(values))
        second_best_idx = values.index(
            max([values[i] for i in range(len(values)) if i != best_idx]))

        if values[best_idx] > best_value:
            x_best = pop[best_idx]
            best_value = values[best_idx]

        parent1 = pop[best_idx]
        parent2 = pop[second_best_idx]
        x_cross, a = croisement(parent1, parent2, cost, iteration)

        x_mut = mutation(x_cross, proba_pertubation)

        worst_idx = values.index(min(values))
        pop[worst_idx] = x_mut

    ##prendre une sol realisable
    valeurs = [fit(x, a, b, cost) for x in pop]
    index_sorted = sorted(range(len(valeurs)), key = lambda i:valeurs[i])
    method_real='1'
    if method_real=='1': # prendre la première solution réalisable
        pop[index_sorted[0]] = reparation(pop[index_sorted[0]], a, b, cost)  ##éviter la boucle infinie
        while not is_realisable(pop[index_sorted[-1]], a, b): ##prendre la meilleure solution réalisable
            index_sorted = index_sorted[:-1]
        x = pop[index_sorted[-1]]

    else:
        if not is_realisable(pop[index_sorted[-1]], a, b):
            pop[index_sorted[-1]] = reparation(pop[index_sorted[-1]], a, b, cost) ##réparer la meilleure solution
            x= pop[index_sorted[-1]]
        else:
            x = pop[index_sorted[-1]]
    value = np.dot(cost, x)
   
    # Génération du graphe des statistiques
    plt.figure(figsize=(10, 6))
    plt.plot(stats['iteration'], stats['max_fitness'], label='Max Fitness', linewidth=2)
    plt.plot(stats['iteration'], stats['mean_fitness'], label='Mean Fitness', linewidth=2)
    plt.plot(stats['iteration'], stats['best_realisable'], label='Best Realisable Fitness', linewidth=2, linestyle='--')
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

    return x, value