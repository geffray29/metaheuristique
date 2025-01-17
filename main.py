##metaheuristique
import numpy as np
import random
import requests
import functions

if __name__ == '__main__':
    # Charger les données
    mon_fichier = "mknapcb3.txt"
    nb_instances, instances = functions.extract_data(mon_fichier)
    
    # Sélectionner une instance spécifique pour l'algorithme génétique
    num_instance = 20
    a = instances[num_instance]["A"]
    b = instances[num_instance]["B"]
    cost = instances[num_instance]["cost"]
    n = instances[num_instance]["Nb projet"]
    m = instances[num_instance]["Nb ressource"]
    val_opt = instances[num_instance]["Valeur optimale"]

    # Paramètres de l'algorithme génétique
    nb_iter = 80
    taille_pop = 200
    max_pop = 150
    taux_mut = 0.1
    proba_pertubation = 0.5

    # Exécuter l'algorithme génétique
    sol_init = '0'
    init = []
    #real = []
  
    xgen, value_gen = functions.algorithme_genetique_stats(
    n, m, cost, a, b, nb_iter, taille_pop, max_pop, taux_mut, 
    functions.heuristique_sac_a_dos, functions.voisinage, sol_init, proba_pertubation)
    init.append(value_gen)

    x_heur, value_heur = functions.heuristique_sac_a_dos(n, m, cost, a, b, '0')
    x_mont, value_monte = functions.algorithme_montee(x_heur, a, b, cost, functions.voisinage)
    print('sol heur', value_heur)
    print('sol monte', value_monte)
    print('sol init from mont', init)
    print('valopt', val_opt)
    #print('realisable', real)

    pass