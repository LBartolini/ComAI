import numpy as np
from Net import Network

def genetic_change(bests, n_total):
    new_nets = []
    n_bests = len(bests)
    weight_dim = len(bests[0])

    # modified bests
    for i, j in zip(np.logspace(-5, -1, n_bests), range(n_bests)):
        new_nets.append(bests[j] + (np.random.uniform(-1, 1, weight_dim)*i)*bests[j])

    # combined bests
    part_of_best = int((weight_dim / n_bests))
    remainder = weight_dim % n_bests
    for _ in range(n_bests):
        tmp = []
        for i in range(n_bests):
            tmp += bests[i][i*part_of_best:i*part_of_best+part_of_best]
        tmp += bests[0][-remainder:]
        new_nets.append(tmp)

    #same exact bests
    for i in range(n_bests):
        new_nets.append(bests[i])

    #random weights to match n_total
    for _ in range(n_total-len(new_nets)):
        new_nets.append(np.random.uniform(-1, 1, weight_dim))

    return np.array(new_nets)

def find_res(v, n):
    for g in v:
        if g.forward_propagation(np.array([0.4]))[0] > n:
            return True, g.export(), g
    return False, False

def find_bests(ginos, n_bests):
    results = {}
    for g in ginos:
        results[g.forward_propagation(np.array([0.4]))[0]] = g.export()
    
    ret = []
    for key in sorted(results)[-n_bests:]:
       ret.append(list(results[key]))

    return ret

k = 0
n = 0.999
pop = 3000
n_bests = 800

ginos = [Network([1, 5, 1]) for _ in range(pop)]

while not find_res(ginos, n)[0]:
    k += 1
    bests = find_bests(ginos, n_bests)
    for i, vect in enumerate(genetic_change(bests, pop)):
        ginos[i]._import(vect)

tmp = find_res(ginos, n)
print(tmp, tmp[2].forward_propagation(np.array([0.4])), k)