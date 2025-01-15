
##metaheuristique
import numpy as np
import random
import requests
import functions

if __name__ == '__main__':
    # Charger les données
    mon_fichier = "mknapcb1.txt"
    nb_instances, instances = functions.extract_data(mon_fichier)
    
    # Sélectionner une instance spécifique pour l'algorithme génétique
    num_instance = 4
    a = instances[num_instance]["A"]
    b = instances[num_instance]["B"]
    cost = instances[num_instance]["cost"]
    n = instances[num_instance]["Nb projet"]
    m = instances[num_instance]["Nb ressource"]
    val_opt = instances[num_instance]["Valeur optimale"]

    # Paramètres de l'algorithme génétique
    nb_iter = 180
    taille_pop = 200
    max_pop = 150
    taux_mut = 0.1

    # Exécuter l'algorithme génétique
    xgen, value_gen = functions.algorithme_genetique(
        n, m, cost, a, b, nb_iter, taille_pop, max_pop, taux_mut, 
        functions.heuristique_sac_a_dos, functions.voisinage
    )

    # Afficher les résultats
    print(f"Resultats de l'algorithme genetique pour l'instance {num_instance}:")
    print(f"Valeur optimale : {val_opt}")
    print(f"Valeur obtenue : {value_gen}")
    print(f"Solution : {xgen}")
    pass