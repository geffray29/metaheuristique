import numpy as np
def reparation(x, a, b, cost):
    """
    Algorithme de réparation avec suppression et ajout.
    
    Paramètres :
    - x : Solution initiale (vecteur binaire).
    - a : Matrice (M x N) des consommations de ressources.
    - b : Liste des quantités disponibles de chaque ressource.
    - cost : Liste des coûts associés à chaque projet.

    Retourne :
    - x : Solution réparée (vecteur binaire).
    - gain : Gain total de la solution réparée.
    """
    # Calcul des ressources totales disponibles
    b_prime = np.sum(b) 
    a_prime = np.sum(a, axis=0)  # Somme des consommations par projet

    # Phase de suppression : Respecter les contraintes
    y = -cost / a_prime  # Ratio coût par ressource consommée
    indices = y.argsort()  # Trier par coût croissant
    index_to_pop = [i for i in indices if x[i] == 1]

    # Tant que les contraintes sont violées, supprimez les projets de faible priorité
    while any(np.dot(a, x) > b) and index_to_pop:
        last_index = index_to_pop[-1]
        x[last_index] = 0
        index_to_pop = index_to_pop[:-1]

    # Phase d'ajout : Maximiser les gains
    gain_ratios = cost / a_prime  # Ratio gain par ressource consommée
    indices_to_add = gain_ratios.argsort()[::-1]  # Trier par gain décroissant

    # Tenter d'ajouter des projets viables
    for i in indices_to_add:
        if x[i] == 0 and all(np.dot(a, x) + a[:, i] <= b):
            x[i] = 1
    return x