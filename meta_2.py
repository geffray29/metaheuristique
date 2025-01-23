from utils_2 import *
import numpy as np
import requests
import random
import matplotlib.pyplot as plt
import random
from meta_genetique import *



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
    print(is_feasible(x_best, a, b))
    best_value = np.dot(x_best, cost)
    return x_best, best_value