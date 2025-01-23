import numpy as np
def is_realisable(x, a, b):
    return all(np.dot(a, x) <= b)

## 1 bit flip
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

## ce voisinage permutte deux bits de x
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

## renvoie un voisinage réalisable
def voisin_realisable(x, a, b, fct_voisinage):
    voisins = fct_voisinage(x)
    voisins_realisables = []
    for voisin in voisins:
        if is_realisable(voisin, a, b):
            voisins_realisables.append(voisin)
    return voisins_realisables