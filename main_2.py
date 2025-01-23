
import numpy as np
import random
import requests
import functions
import utils_2 as utils
import meta_2 as meta

if __name__ == '__main__':
    # Charger les données
    mon_fichier = "mknapcb1.txt"
    nb_instances, instances = utils.extract_data(mon_fichier)
    
    # Sélectionner une instance spécifique pour l'algorithme génétique
    num_instance = 4
    a = instances[num_instance]["A"]
    b = instances[num_instance]["B"]
    cost = instances[num_instance]["cost"]
    n = instances[num_instance]["Nb projet"]
    m = instances[num_instance]["Nb ressource"]
    val_opt = instances[num_instance]["Valeur optimale"]

    # Paramètres de l'algorithme génétique
    nb_iter = 350
    taille_pop = 150
    max_pop = 100
    taux_mut = 0.1
    proba_pertubation = 0.3

    # Exécuter l'algorithme génétique
    sol_init = 'random'
    
    #real = []
  
    xgen, value_gen = meta.genetic_algo(
    n, m, cost, a, b, nb_iter, taille_pop, max_pop, taux_mut, 
    utils.heuristique_sac_a_dos, utils.voisinage, sol_init, proba_pertubation)
    

    x_heur, value_heur = utils.heuristique_sac_a_dos(n, m, cost, a, b, '0')
    x_mont, value_monte = utils.algorithme_montee(x_heur, a, b, cost, functions.voisinage)
    print('sol heur', value_heur)
    print('sol monte', value_monte)
    print('sol init from mont', value_gen)
    print('valopt', val_opt)
    #print('realisable', real)

    pass