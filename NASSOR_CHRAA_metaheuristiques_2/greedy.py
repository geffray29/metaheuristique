import numpy as np
from instances import get_instances


def greedy_knapsack(c, a, b):
    """ Heuristique gloutonne pour le problème du sac à dos multidimensionnel.

    Paramètres :
    c : Liste des gains associés aux projets.
    a : Matrice (M x N) des consommations de ressources.
    b : Liste des quantités disponibles de chaque ressource.

    Retourne :
    x : Solution binaire (0 ou 1) indiquant les projets sélectionnés.
    """

    N = len(c)
    M = len(b)

    ratios = np.array(c) / (np.sum(a, axis=0) + 1e-6)
    indices = np.argsort(-ratios)

    x = np.zeros(N, dtype=int)
    resource_usage = np.zeros(M)

    for j in indices:
        if np.all(resource_usage + a[:, j] <= b):
            x[j] = 1
            resource_usage += a[:, j]
    value_solution = np.sum(c * x)

    return x, value_solution


# x, value_solution = greedy_knapsack(c, a, b)
# print(f"Solution : {x}")
# print(f"Valeur de la solution : {value_solution}")
# print(f"Valeur optimale : {inst['opt_value']}")
