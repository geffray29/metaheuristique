import random
from instances import get_instances
import numpy as np
from heuristique_de_reparation import reparation, gen_sol_initiale
from structure_de_voisinage import gen_voisin_perm, hamming_1, hamming_2
from tqdm import tqdm
import pandas as pd

random.seed(2)  # inst1_5


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
            if is_feasible(x_vois, a, b):
                sols.append(x_vois)
                iters += 1
                if iters == n_sols:
                    break
    return sols

    

def is_feasible(x, a, b):
    """Vérifie si une solution est faisable."""
    return np.all(np.dot(a, x) <= b)


def mutate_solution(N, M, a, b, c, x):
    
    x_mut = x.copy()
    i = random.randint(0, len(x)-1)
    x_mut[i] = 1 - x_mut[i]
    x_mut = reparation(N, M, a, b, c, x_mut)[0]
    return x_mut


def cross_over(x1, x2,N, M, a, b, c):
    x_cross = x1.copy()
    i = random.randint(0, len(x1)-1)
    x_cross[i:] = x2[i:]
    x_cross = reparation(N, M, a, b, c, x_cross)[0]
    return x_cross


def roue_fortune_selection(population, values):

    # Calcul des probabilités cumulées
    total_fitness = sum(values)
    if total_fitness == 0:  # Évite la division par zéro
        return random.choice(population)

    probabilities = [value / total_fitness for value in values]
    cumulative_probabilities = np.cumsum(probabilities)

    # Générer un nombre aléatoire entre 0 et 1
    random_pick = random.uniform(0, 1)

    # Trouver l'individu correspondant
    for i, cumulative_probability in enumerate(cumulative_probabilities):
        if random_pick <= cumulative_probability:
            return population[i]

    # Au cas où (ne devrait pas se produire)
    return population[-1]


def genetic_algo(N, M, c, a, b, max_iter=100, pop_size=100):
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
    time = 0
    N = len(c)
    M = len(b)
    population = gen_feasible_sols(N, M, a, b, c, pop_size)
    best_value = 0
    x_best = population[0]

    for _ in range(max_iter):
        values = [sum(c[j] * x[j] for j in range(N)) for x in population]

        best_idx = values.index(max(values))
        second_best_idx = values.index(
            max([values[i] for i in range(pop_size) if i != best_idx]))

        if values[best_idx] > best_value:
            x_best = population[best_idx]
            best_value = values[best_idx]

        # x1 = population[best_idx]
        # x2 = population[second_best_idx]
        x1 = roue_fortune_selection(population, values)
        x2 = roue_fortune_selection(population, values)
        x_cross = cross_over(x1, x2,N, M, a, b, c)

        x_mut = mutate_solution(N, M, a, b, c, x_cross)

        worst_idx = values.index(min(values))
        population[worst_idx] = x_mut
        # print(f"Nouvelle solution trouvée :{best_value} : {is_feasible(x_best, a, b)}")

    return x_best, best_value


# files = ['instances/mknap1']
# df = pd.DataFrame(columns=['file','instance', 'opt_value','value_genetic', 'gap_genetic'])
# with tqdm(total=len(files), desc="Traitement des fichiers", unit="fichier") as overall_progress:
#     for file in files : 
#         inst_file = file + '.txt'
        
#         instances = get_instances(inst_file)
#         with tqdm(total=len(instances), desc=f"Traitement de {file}", leave=True, unit="instance") as file_progress:
#             for i in range(len(instances)):
#                 ins = instances[i]
                
#                 c = ins["gains"]
#                 a = np.array(ins["ressources"])
#                 b = ins["quantite_ressources"]
#                 N = len(c)
#                 M = len(b)
#                 x_best, best_value = genetic_algo(N, M, c, a, b, max_iter=200, pop_size=100)
#                 infos = infos = {
#                     'file': [file.split('/')[-1]],
#                     'instance': [i],
#                     'opt_value': [float(ins["opt_value"])],
#                     'value_genetic': [best_value],
#                     'gap_genetic': [(float(ins["opt_value"] - best_value)) / float(ins["opt_value"])],
#                 }
#                 df = pd.concat([df, pd.DataFrame(infos)])
#                 file_progress.update(1)
#         overall_progress.update(1)
        
# file = 'mknap1'
# caption = "{"+ f"Résultats pour l'instance {file}" + "}"
# print( "\\begin{table}[]\n\centering")
# print(df.query("file == @file").to_latex(index=False, formatters={"name": str.upper},
#                   float_format="{:.3f}".format,))
# print(f"\caption{caption}")
# print("\label{tab:my_label}")
# print("\end{table}")       
