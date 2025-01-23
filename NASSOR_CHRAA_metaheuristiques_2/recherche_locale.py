import numpy as np
from instances import get_instances
from greedy import greedy_knapsack
from structure_de_voisinage import hamming_1, hamming_2
from heuristique_de_reparation import reparation_surrogate



def is_feasible(x, a, b):
    """Vérifie si une solution est faisable."""
    return np.all(np.dot(a, x) <= b)


def local_search(N, M, c, a, b, x_init, voisinage='hamming1'):
    """ Méthode de recherche locale pour le problème du sac à dos multidimensionnel.

    Paramètres :
    c : Liste des gains associés aux projets.
    a : Matrice (M x N) des consommations de ressources.
    b : Liste des quantités disponibles de chaque ressource.
    x_init : Solution initiale (vecteur binaire).
    voisinage : Structure de voisinage:
                -> hamming1 : distance de Hamming 1
                -> hamming2 : distance de Hamming 2

    Retourne :
    x_best : Meilleure solution trouvée.
    best_value : Gain total associé à la meilleure solution.
    """

    # Initialisation
    x_best = x_init.copy()
    best_value = np.dot(c, x_best)
    improved = True

    while improved:
        improved = False
        if voisinage == 'hamming1':
            neighbors = hamming_1(x_best)
        elif voisinage == 'hamming2':
            neighbors = hamming_2(x_best)
        else:
            raise ValueError("Voisinage invalide")
        for x_neighbor in neighbors:
            if is_feasible(x_neighbor, a, b):
                neighbor_value = np.dot(c, x_neighbor)
                if neighbor_value > best_value:
                    #print("Nouvelle solution trouvée :", neighbor_value)
                    x_best = x_neighbor
                    best_value = neighbor_value
                    improved = True
                    break

    return x_best, best_value

def recherche_locale_combinee(N, M, c, a, b, x_init):
    """ Méthode de recherche locale qui combine 2 structures de voisinage pour le problème du sac à dos multidimensionnel.

    Paramètres :
    c : Liste des gains associés aux projets.
    a : Matrice (M x N) des consommations de ressources.
    b : Liste des quantités disponibles de chaque ressource.
    x_init : Solution initiale (vecteur binaire).
    
    Retourne :
    x_best : Meilleure solution trouvée.
    best_value : Gain total associé à la meilleure solution.
    """
    # Initialisation
    x_best = x_init.copy()
    best_value = np.dot(c, x_best)
    improved = True

    while improved:
        improved = False
        for x_neighbor in hamming_1(x_best):
            if is_feasible(x_neighbor, a, b):
                neighbor_value = np.dot(c, x_neighbor)
                if neighbor_value > best_value:
                    x_best = x_neighbor
                    best_value = neighbor_value
                    improved = True
                    break # On retourne aux voisins Hamming 1 sur la nouvelle solution

        if improved:
            continue # On saute le reste du code et on passe à l'itération suivante
        
        # Voisinage Hamming 2

        for x_neighbor in hamming_2(x_best):
            if is_feasible(x_neighbor, a, b):
                neighbor_value = np.dot(c, x_neighbor)
                if neighbor_value > best_value:
                    x_best = x_neighbor
                    best_value = neighbor_value
                    improved = True
                    break # On retourne aux voisins Hamming 1 sur la nouvelle solution

    return x_best, best_value



#file_name = "instances/mknapcb4.txt"

#instances = get_instances(file_name)
#inst = instances[0]

#c = inst["gains"]
#a = np.array(inst["ressources"])
#b = inst["quantite_ressources"]
#N = int(inst["nb_projets"])
#M = int(inst["nb_sacs"])

#x_init, value_solution_init = greedy_knapsack(c, a, b)

#x_best, best_value = recherche_locale_combinee(N, M, c, a, b, x_init)

#print("Solution initiale :", x_init)
#print("Meilleure solution trouvée :", x_best)
#print("Valeur de la solution initiale g1:", value_solution_init)
#print("Nouvelle solution g1:", best_value)
#print("Valeur optimale :", inst['opt_value'])

#x_init2, value_solution_init2 = reparation_surrogate(N, M, a, b, c, methode='simple')

#x_best2, best_value2 = local_search(N, M, c, a, b, x_init2, 'hamming1')

#print("Solution initiale :", x_init2)
#print("Meilleure solution trouvée :", x_best2)
#print("Valeur de la solution initiale r1:", value_solution_init2)
#print("Nouvelle solution r1:", best_value2)
#print("Valeur optimale :", inst['opt_value'])

#x_init3, value_solution_init3= greedy_knapsack(c, a, b)

#x_best3, best_value3 = recherche_locale_combinee(N, M, c, a, b, x_init3)

#print("Solution initiale :", x_init3)
#print("Meilleure solution trouvée :", x_best3)
#print("Valeur de la solution initiale g2:", value_solution_init3)
#print("Nouvelle solution g2:", best_value3)
#print("Valeur optimale :", inst['opt_value'])

#x_init4, value_solution_init4 = reparation_surrogate(N, M, a, b, c, methode='simple')

#x_best4, best_value4 = recherche_locale_combinee(N, M, c, a, b, x_init4)

#print("Solution initiale sur :", x_init4)
#print("Meilleure solution trouvée :", x_best4)
#print("Valeur de la solution initiale r2:", value_solution_init4)
#print("Nouvelle solution r2:", best_value4)
#print("Valeur optimale :", inst['opt_value'])