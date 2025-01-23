def read_instance(inst, nb_projets, nb_sacs, opt_value):
    return {
        "nb_projets": float(nb_projets),
        "nb_sacs": float(nb_sacs),
        "opt_value": float(opt_value),
        "gains": [float(gain) for gain in inst[3:3+int(nb_projets)]],
        "ressources": [[float(ressource) for ressource in inst[3+int(nb_projets) + i*int(nb_projets): 3+int(nb_projets) + i*int(nb_projets) + int(nb_projets)]] for i in range(0, int(nb_sacs))],
        "quantite_ressources": [float(quant) for quant in inst[3+int(nb_projets) + int(nb_sacs)*int(nb_projets):]]
    }


def get_instances(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        data = content.split()
        nb_instances = data[0]
        instances_brut = data[1:]
        instances = []
        for i in range(int(nb_instances)):
            nb_projects = int(instances_brut[0])
            nb_sacs = int(instances_brut[1])
            opt_value = float(instances_brut[2])
            inst = instances_brut[:3+nb_projects +
                                  nb_sacs + nb_projects*nb_sacs]
            instances.append(read_instance(
                inst, nb_projects, nb_sacs, opt_value))
            instances_brut = instances_brut[3 +
                                            nb_projects + nb_sacs + nb_projects*nb_sacs:]
    return instances
