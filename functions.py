import numpy as np
import requests
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
        index.pop()
    #print("x after dot (a,x) ", x)
    value = np.dot(cost, x)

    return x, value


def is_realisable(x, a, b):
    return np.all(np.dot(a, x) <= b)


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


def perturber_solution(x):
    n = x.size
    for _ in range(np.random.randint(1, 2+ n//10)):
        index = np.random.randint(len(x))
        x[index] = 1 - x[index]
    return x


def fit(x, a, b, cost):
    if not is_realisable(x, a, b):
        return -1
    return np.dot(cost, x)

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

'''
def reparation_v2(x, a, b, cost):
    b_prime = np.sum(b) 
    a_prime= np.sum(a, axis=0) 
    y = -cost/a_prime 
    indices =y.argsort()
    index_to_pop = [i for i in indices if x[i] == 1]
    while any(np.dot(a, x) > b):
        last_index = index_to_pop[-1]
        x[last_index] = 0
        #print(x)
        index_to_pop.pop()
    return x
'''

def hamming_1(x):
    """Génère tous les voisins à une distance de Hamming = 1."""
    neighbors = []
    for i in range(len(x)):
        neighbor = x.copy()
        neighbor[i] = 1 - neighbor[i]  # Inverse le bit
        neighbors.append(neighbor)
    return neighbors

def gen_sol_initiale(N):
    """ Génère une solution initiale binaire aléatoire. 
    Paramètres :
    N : Nombre de projets
    Retourne :
    x : Solution initiale binaire (0 ou 1) indiquant les projets sélectionnés.
    """
    return [random.randint(0, 1) for _ in range(N)]

def gen_voisin_perm(N, M, a, b, c, x, max_iter=100):
    """ Génère une solution voisine, par séléction et permutation d'un projet sélectionné et 
    d'un autre non séléctionné, pour le problème du sac à dos multidimensionnel.
    """

    # Tableau des indices des projets selectionnés dans la solution x
    selec = [i for i, xi in enumerate(x) if xi == 1]
    # Tableau des indices des projets non selectionnés dans la solution x
    non_selec = [i for i, xi in enumerate(x) if xi == 0]

    for _ in range(max_iter):

        # Choix aléatoire d'un projet selectionné
        p_selec = random.choice(selec)
        # Choix aléatoire d'un projet non selectionné
        p_non_selec = random.choice(non_selec)

        x_vois = x[:]

        # permutation des deux projets

        x_vois[p_selec] = 0
        x_vois[p_non_selec] = 1

        if is_realisable(x_vois, a, b):
            return x_vois

    return x

def gen_random_sols(N, M, a, b, c, n_sols):
    """Fonction qui génère n_sols solutions aléatoires pour le problème.
    On génère une solution initiale aléatoire, on la répare et on génère n_sols-1 solutions en explorant les voisins en utilsant la permutaion.
    """
    first_rd_sol = gen_sol_initiale(N)
    feasible_sol, _ = reparation(N, M, a, b, c, first_rd_sol)

    # On génère n_sols solutions aléatoires on explorant les voisinages
    sols = [feasible_sol]
    for _ in range(n_sols-1):
        x = feasible_sol.copy()
        for _ in range(10):
            x = gen_voisin_perm(N, M, a, b, c, x)
        sols.append(x)

    return sols
def gen_feasible_sols(N, M, a, b, c, n_sols):
    """Fonction qui génère n_sols solutions aléatoires pour le problème. On génère une solution initiale aléatoire, on la répare et on génère n_sols-1 solutions en explorant les voisins avec la methode de la distance de Hamming.
    """
    first_rd_sol = gen_sol_initiale(N)
    feasible_sol, _ = reparation(N, M, a, b, c, first_rd_sol)
    
    sols = [feasible_sol]
    iters = 0
    while iters < n_sols:
        x = feasible_sol.copy()
        x = hamming_1(x)
        # if soltion is feasible add it to the list
        for x_vois in x:
            if is_realisable(x_vois, a, b):
                sols.append(x_vois)
                iters += 1
                if iters == n_sols:
                    break
    return sols
'''
def generation_pop(n, m, cost, a, b, taille_pop, generation_solution, fct_voisinage):   
    x = generation_solution(n, m, cost, a, b, fct_voisinage)[0]
    #x = [int(i) for i in x]
    popu = [x]
    for _ in range (taille_pop-1):
        x = perturber_solution(x)
        popu.append(x)
    return popu
'''

def algorithme_genetique(n, m, cost, a, b, nb_iter, taille_pop, max_pop, taux_mut, generation_solution, fct_voisinage):
    max_pop = (max_pop//2)*2
    #population initiale
    pop = gen_feasible_sols(n, m, a, b, cost, taille_pop)
    random.shuffle(pop)
    #iterations
    for _ in range(nb_iter):
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
            #print(len(parents[i]))
            #print(len(parents[i+1]))
            child1 = np.concatenate((parents[i][:cut], parents[i+1][cut:]))
            child2 = np.concatenate((parents[i+1][:cut], parents[i][cut:]))
            #print('x',child2.shape)
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
            #print('x',child2.shape)
        for j in range(len(child)):
            if not is_realisable(child[j], a, b):
                child[j] = reparation(n, m, a, b, cost, child[j])
        #mutation
        for j in range(len(child)):
            if np.random.rand() < taux_mut:
                child[j] = perturber_solution(child[j])
        pop = parents + child
        random.shuffle(pop)
    valeurs = [fit(x, a, b, cost) for x in pop]
    index = valeurs.index(max(valeurs))
    x= pop[index]
    value = np.dot(cost, pop[index])
    return x, value
