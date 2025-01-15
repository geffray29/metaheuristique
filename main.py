##metaheuristique
import numpy as np
import requests
import random
import functions

##TELECHARGEMENT DATA

base_url = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb"

if __name__ == '__main__':
    for i in range(9):
        url = base_url + str(i+1) + ".txt"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Vérifie qu'il n'y a pas d'erreur HTTP
            contenu_texte = response.text

            with open("mknapcb" + str(i+1) + ".txt", "w") as fichier:
                fichier.write(contenu_texte)


        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de l'ouverture de l'URL : {e}")

    ## EXTRACTION DATA


    ## VISUALISATION DATA

    mon_fichier = "C:/Users/Thibaud/3A/metaheuristique/Data/mknap1.txt"
    nb_instances, instances = functions.extract_data(mon_fichier)

    print(nb_instances)
    print(instances)

    ## HEURISTIQUE

    ##TEST HEURISTIQUE

    mon_fichier = "C:/Users/Thibaud/3A/metaheuristique/Data/mknap1.txt"
    nb_instances, instances = functions.extract_data(mon_fichier)

    for i in range(1,nb_instances+1):
        a = instances[i]["A"]
        b = instances[i]["B"]
        cost = instances[i]["cost"]
        n = instances[i]["Nb projet"]
        m = instances[i]["Nb ressource"]
        val_opt = instances[i]["Valeur optimale"]
        x, value =functions.heuristique_sac_a_dos(n, m, cost, a, b,1)
        print('val_opt=', val_opt, 'value=', value, x)

    chemin = "C:/Users/Thibaud/3A/metaheuristique/Data/result.txt"

    with open(chemin, "w") as fichier:
        fichier.write("RESULT\n\n")

    ## TELECHARGEMENT SOLUTIONS HEURISTIQUE GLOUTONNE


    mon_fichier = "C:/Users/Thibaud/3A/metaheuristique/Data/mknap1.txt"
    nb_instances, instances = functions.extract_data(mon_fichier)
    with open(chemin, "a", encoding="utf-8") as f:
        f.write(f"FICHIER : mknap1 \n\n")
    for j in range(1,nb_instances+1):
        a = instances[j]["A"]
        b = instances[j]["B"]
        cost = instances[j]["cost"]
        n = instances[j]["Nb projet"]
        m = instances[j]["Nb ressource"]
        val_opt = instances[j]["Valeur optimale"]
        x, value =functions.heuristique_sac_a_dos(n, m, cost, a, b, fct_voisinage=1)
        chemin = "C:/Users/Thibaud/3A/metaheuristique/Data/result.txt"
        with open(chemin, "a", encoding="utf-8") as f:
            f.write(f"instance {j} : {value} \n")
            f.write(f"optimal value : {val_opt} \n")
            f.write(f"gap : {(val_opt-value)/val_opt} \n")
            f.write(f"projet selectionné : {x} \n")
            f.write("\n")
        

    ## TELECHARGEMENT SOLUTIONS HEURISTIQUE GLOUTONNE

    base_fichier = "C:/Users/Thibaud/3A/metaheuristique/Data/mknapcb"

    for i in range(9):
        mon_fichier = base_fichier + str(i+1) + ".txt"
        nb_instances, instances = functions.extract_data(mon_fichier)
        with open(chemin, "a", encoding="utf-8") as f:
            f.write(f"FICHIER : mknapcb{i+1} \n\n")
        for j in range(1,nb_instances+1):
            a = instances[j]["A"]
            b = instances[j]["B"]
            cost = instances[j]["cost"]
            n = instances[j]["Nb projet"]
            m = instances[j]["Nb ressource"]
            val_opt = instances[j]["Valeur optimale"]
            x, value =functions.heuristique_sac_a_dos(n, m, cost, a, b, fct_voisinage=1)
            chemin = "C:/Users/Thibaud/3A/metaheuristique/Data/result.txt"
            with open(chemin, "a", encoding="utf-8") as f:
                f.write(f"instance {j} : {value} \n")
                f.write(f"optimal value : {val_opt} \n")
                f.write(f"gap : {(val_opt-value)/val_opt} \n")
                f.write(f"projet selectionné : {x} \n")
                f.write("\n")


    ## DEFINITION D'UN VOISINAGE





    ## ALGORITHME DE MONTEE




    ##TEST ALGORITHME DE MONTEE

    mon_fichier = "C:/Users/Thibaud/3A/metaheuristique/Data/mknap1.txt"
    nb_instances, instances = functions.extract_data(mon_fichier)
    num_instance = 5
    a = instances[num_instance]["A"]
    b = instances[num_instance]["B"]
    cost = instances[num_instance]["cost"]
    n = instances[num_instance]["Nb projet"]
    m = instances[num_instance]["Nb ressource"]
    val_opt = instances[num_instance]["Valeur optimale"]
    x, value =functions.heuristique_sac_a_dos(n, m, cost, a, b, fct_voisinage=1)
    xmax, value_monte = functions.algorithme_montee(x, a, b, cost, functions.voisinage)
    print('val_opt=', val_opt, 'value=', value, 'value_monte=', value_monte)
    print('x_type', type(x))
    print('xmax_type', type(xmax))


    ## AFFICHAGE SOLUTION ALGORITHME DE MONTEE

    mon_fichier = "C:/Users/Thibaud/3A/metaheuristique/Data/mknap1.txt"
    nb_instances, instances = functions.extract_data(mon_fichier)

    L_value=[]
    L_value_monte=[]
    L_value_opt=[]

    for i in range(1,nb_instances+1):
        a = instances[i]["A"]
        b = instances[i]["B"]
        cost = instances[i]["cost"]
        n = instances[i]["Nb projet"]
        m = instances[i]["Nb ressource"]
        val_opt = instances[i]["Valeur optimale"]
        x, value =functions.heuristique_sac_a_dos(n, m, cost, a, b, fct_voisinage=1)
        xmax, value_monte = functions.algorithme_montee(x, a, b, cost,functions.voisinage)
        L_value.append(value)
        L_value_monte.append(value_monte)
        L_value_opt.append(val_opt)
        print('val_opt=', val_opt,  'value_monte=', value_monte,'value_heuristique=', value)



    ## DEUXIEME VOISINAGE : échange de projet

    mon_fichier = "C:/Users/Thibaud/3A/metaheuristique/Data/mknapcb1.txt"
    nb_instances, instances = functions.extract_data(mon_fichier)

    L_value=[]
    L_value_monte=[]
    L_value_opt=[]

    for i in range(1,nb_instances+1):
        a = instances[i]["A"]
        b = instances[i]["B"]
        cost = instances[i]["cost"]
        n = instances[i]["Nb projet"]
        m = instances[i]["Nb ressource"]
        val_opt = instances[i]["Valeur optimale"]
        x, value =functions.heuristique_sac_a_dos(n, m, cost, a, b, fct_voisinage=1)
        xmax, value_monte = functions.algorithme_montee(x, a, b, cost, functions.voisinage_echange)
        L_value.append(value)
        L_value_monte.append(value_monte)
        L_value_opt.append(val_opt)
        print('val_opt=', val_opt,  'value_monte=', value_monte,'value_heuristique=', value)



    ## ALOGORITHE GENETIQUE



    ##TEST ALGORITHME GENETIQUE

    mon_fichier = "C:/Users/Thibaud/3A/metaheuristique/Data/mknapcb1.txt"

    nb_instances, instances = functions.extract_data(mon_fichier)
    num_instance = 4
    a = instances[num_instance]["A"]
    b = instances[num_instance]["B"]
    cost = instances[num_instance]["cost"]
    n = instances[num_instance]["Nb projet"]
    m = instances[num_instance]["Nb ressource"]
    val_opt = instances[num_instance]["Valeur optimale"]
    x, value =functions.heuristique_sac_a_dos(n, m, cost, a, b, fct_voisinage=1)
    xmax, value_monte = functions.algorithme_montee(x, a, b, cost, functions.voisinage)
    xgen, value_gen = functions.algorithme_genetique(n, m, cost, a, b, 180, 200, 150, 0.1, functions.heuristique_sac_a_dos, functions.voisinage)
    print('val_opt=', val_opt, 'value=', value, 'value_monte=', value_monte, 'value_gen=', value_gen)

    pass