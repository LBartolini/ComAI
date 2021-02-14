import numpy as np
from Net import Network
from Agent import Agent
import random


def genetic_change(bests, n_total):
    new_nets = []
    n_bests = len(bests)
    weight_dim = len(bests[0])

    # modified bests
    for i, j in zip(np.logspace(-5, -1, n_bests), range(n_bests)):
        new_nets.append(
            bests[j] + (np.random.uniform(-1, 1, weight_dim)*i)*bests[j])

    # combined bests
    part_of_best = int((weight_dim / n_bests))
    remainder = weight_dim % n_bests
    for _ in range(n_bests):
        tmp = []
        for i in range(n_bests):
            tmp += bests[i][i*part_of_best:i*part_of_best+part_of_best]
        tmp += bests[0][-remainder:]
        new_nets.append(tmp)

    # same exact bests
    for i in range(n_bests):
        new_nets.append(bests[i])

    # random weights to match n_total
    for _ in range(n_total-len(new_nets)):
        new_nets.append(np.random.uniform(-1, 1, weight_dim))

    return np.array(new_nets, dtype=object)


def check_sleeping(population):
    for agent in population:
        agent.check_sleeping()


def get_ready(population):
    ret = []
    for agent in population:
        if agent.is_ready():
            ret.append(agent)

    return ret


def get_remaining_ready(ready, speakers):
    ret = []
    for agent in ready:
        if agent not in speakers:
            ret.append(agent)

    return ret


def assign_actions(speakers, listeners):
    for i, speaker in enumerate(speakers):
        action_requested = np.eye(3)[np.random.randint(0, 3)]
        word = speaker.shout_word(action_requested)
        is_action_done_correct = listeners[i].do_action(word, action_requested)
        if is_action_done_correct:
            speaker.change_score(0.5)
            listeners[i].change_score(1)
        else:
            speaker.change_score(-0.15)
            listeners[i].change_score(-0.25)


def print_best_score(population):
    print(find_bests(population, 1)[0].score)


def find_bests(population, n_best):
    sorted_pop = sorted(population, key=lambda x: x.score, reverse=True)
    return sorted_pop[:n_best]


def make_new_population(population, population_size, n_best):
    best_agents = find_bests(population, n_best)

    best_listeners = [list(ag.hearing.export()) for ag in best_agents]
    best_speakers = [list(ag.speaking.export()) for ag in best_agents]

    new_listeners = genetic_change(best_listeners, population_size)
    new_speakers = genetic_change(best_speakers, population_size)

    for i, weights in enumerate(zip(new_listeners, new_speakers)):
        population[i].reset()
        population[i].hearing._import(weights[0])
        population[i].speaking._import(weights[1])
    
    return population


def game(population_size, n_best, n_word, epochs, turns_per_epoch, turns_to_sleep, verbose=False):
    population = [Agent(n_word, turns_to_sleep) for _ in range(population_size)]

    for epoch in range(epochs):
        print("Epoch ", epoch)
        for _ in range(turns_per_epoch):
            check_sleeping(population)
            ready_agents = get_ready(population)
            speakers = random.sample(ready_agents, int(len(ready_agents)/2))
            listeners = get_remaining_ready(ready_agents, speakers)
            assign_actions(speakers, listeners)

        if verbose:
            print_best_score(population)

        population = make_new_population(population, population_size, n_best)
    
    rnd = random.choice(population)
    word1 = rnd.shout_word([0, 1, 0])
    word2 = rnd.shout_word([1, 0, 0])
    word3 = rnd.shout_word([0, 0, 1])
    print(f"Input: [0, 1, 0] -> Word : {word1}")
    print(f"Word: {word1} -> Action: {rnd.do_action(word1, [0, 1, 0])}\n")
    print(f"Input: [1, 0, 0] -> Word : {word2}")
    print(f"Word: {word2} -> Action: {rnd.do_action(word2, [1, 0, 0])}\n")
    print(f"Input: [0, 0, 1] -> Word : {word3}")
    print(f"Word: {word3} -> Action: {rnd.do_action(word3, [0, 0, 1])}\n")


if __name__ == '__main__':
    def _find_res(v, n):
        for g in v:
            if g.forward_propagation(np.array([0.4]))[0] > n:
                return True, g.export(), g
        return False, False

    def _find_bests(ginos, n_bests):
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

    while not _find_res(ginos, n)[0]:
        k += 1
        bests = _find_bests(ginos, n_bests)
        for i, vect in enumerate(genetic_change(bests, pop)):
            ginos[i]._import(vect)

    tmp = _find_res(ginos, n)
    print(tmp, tmp[2].forward_propagation(np.array([0.4])), k)
