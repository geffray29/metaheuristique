import numpy as np
import random
from instances import get_instances


def gen_sol_initiale(N):
    """ Génère une solution initiale binaire aléatoire. 
    Paramètres :
    N : Nombre de projets
    Retourne :
    x : Solution initiale binaire (0 ou 1) indiquant les projets sélectionnés.
    """
    return [random.randint(0, 1) for _ in range(N)]


def reparation(N, M, a, b, c, x):
    """ Heuristique de réparation pour le problème du sac à dos multidimensionnel.
    Paramètres :
    N : Nombre de projets
    M : Nombre de ressources
    c : Liste des gains associés aux projets.
    a : Matrice (M x N) des consommations de ressources.
    b : Liste des quantités disponibles de chaque ressource.
    x : Solution initiale (vecteur binaire).
    Retourne :
    x : Solution binaire (0 ou 1) indiquant les projets sélectionnés.
    gain : Gain totale de la solution x réparée
    """
    # Calcul de ressouces consommées r[i] pour chaque ressource i
    r = [sum(a[i][j] * x[j] for j in range(N)) for i in range(M)]
    # Phase de supression: le but est de minimiser les pertes lors de la suppression
    while any(r[i] > b[i] for i in range(M)):  # Tant qu'il existe des contraintes violées
        # Calcul de p la liste de priorité pour chaque j tel que x[j] = 1
        p = [(j, c[j] / sum(a[i][j] for i in range(M)))
             for j in range(N) if x[j] == 1]
        p.sort(key=lambda y: y[1])  # tri des priorités par ordre croissant
        j_sup = p[0][0]  # l'indice j à supprimer
        x[j_sup] = 0  # suppression de l'élément j de faible priorité
        for i in range(M):
            r[i] -= a[i][j_sup]
    # Phase d'ajout: le but est de maximiser les gains lors de l'ajout
    # Calcul de e la liste d'efficacité pour chaque j tel que x[j] = 0
    e = [(j, c[j] / sum(a[i][j] for i in range(M)))
         for j in range(N) if x[j] == 0]
    # tri des efficacités par ordre décroissant
    e.sort(key=lambda y: y[1], reverse=True)
    # Ajout des projets x[j]=1 dans la solution tant que cela ne viole pas les contraintes
    for j, _ in e:
        if all(r[i] + a[i][j] <= b[i] for i in range(M)):
            x[j] = 1
            for i in range(M):
                r[i] += a[i][j]
    gain = sum(c[j] * x[j] for j in range(N))
    return x, gain

def reparation_surrogate(N, M, a, b, c, methode='simple'):
    """ Heuristique de réparation après construction d'une solution initiale gloutonne basée sur la relaxation surrogate
    
    Paramètres :
    N : Nombre de projets
    M : Nombre de ressources
    c : Liste des gains associés aux projets.
    a : Matrice (M x N) des consommations de ressources.
    b : Liste des quantités disponibles de chaque ressource.
    methode : - "simple" -> en considérant le vecteur multiplicateur unitaire [1, ..., 1]
              - "inverse" -> en cosidérant le vecteur multiplicateur de l'inverse des capacités [1/b[0], ..., 1/b[M]]
    
    Retourne :
    x : Solution binaire (0 ou 1) indiquant les projets sélectionnés.
    gain : Gain totale de la solution x réparée
    """
    if methode == 'simple':
        u = [1 for _ in range(M)]
    elif methode == 'inverse':
        u = [1 / b[i] for i in range(M)]
    else:
        raise ValueError("Méthode invalide")
    
    # relaxation surrogate 
    
    a_surrogate = [sum(u[i] * a[i][j] for i in range(M)) for j in range(N)]
    b_surrogate = sum(u[i] * b[i] for i in range(M))
     
    # Calcul de e la liste d'efficacité pour chaque j 
    e = [(j, c[j] / (a_surrogate[j] + 1e-6)) for j in range(N)]
    e.sort(key=lambda y: y[1], reverse=True) # tri des efficacités par ordre décroissant
    x = [0 for i in range(N)] # solution initiale
    ressource = 0 # consommation surrogate
    for j, _ in e:
        if ressource + a_surrogate[j] <= b_surrogate:
            x[j] = 1
            ressource += a_surrogate[j]
        
    # Phase de supression des objets de moins bon ratio c[j]/a_surrogate[j] 
    r = [sum(a[i][j] * x[j] for j in range(N)) for i in range(M)]
    while any(r[i] > b[i] for i in range(M)): # Tant qu'il existe des contraintes violées
        p = [(j, c[j] / a_surrogate[j]) for j in range(N) if x[j] == 1] # Calcul de p la liste de priorité pour chaque j tel que x[j] = 1
        p.sort(key=lambda y: y[1]) # tri des priorités par ordre croissant
        j_sup = p[0][0] # l'indice j à supprimer
        x[j_sup] = 0 # suppression de l'élément j de faible priorité 
        for i in range(M):
            r[i] -= a[i][j_sup]
    gain = sum(c[j] * x[j] for j in range(N))
    return x, gain


# file_name = "instances/mknap1.txt"
# instances = get_instances(file_name)
# inst = instances[0]
# print(inst)
# N = int(inst["nb_projets"])
# M = int(inst["nb_sacs"])
# c = inst["gains"]
# a = np.array(inst["ressources"])
# b = inst["quantite_ressources"]
# x = gen_sol_initiale(N)
# x, gain = reparation(N, M, a, b, c, x)
# print(f"Solution : {x}")
# print(f"Valeur de la solution : {gain}")
# print(f"Valeur optimale : {inst['opt_value']}")
