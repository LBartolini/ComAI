from Net import Network, softmax
import numpy as np


def convert_to_onehot(vector):
    softmax_out = softmax(vector)
    argmax = np.argmax(softmax_out)
    ret = np.zeros(len(vector))
    ret[argmax] = 1
    return ret


class Agent():  # aka Gino

    def __init__(self, n_word, turns_to_sleep):
        self.score = 0
        self.turns_to_sleep = turns_to_sleep
        # every turn he goes through +1 when reaching 5 goes to sleep 1 turn
        self.sleep = np.random.randint(0, turns_to_sleep)
        self.now = 0  # 0=ready, 1=sleeping, 2=waiting-other
        # 3 output = 3 possible actions to do
        self.hearing = Network([n_word, 5, 3])
        self.speaking = Network([3, 5, n_word])  # 10 output = 10 words
    
    def reset(self):
        self.score = 0
        self.sleep = np.random.randint(0, self.turns_to_sleep)
        self.now = 0

    def shout_word(self, action_requested):
        out = self.speaking.forward_propagation(action_requested)
        return convert_to_onehot(out)

    def do_action(self, word, action_requested):
        result = convert_to_onehot(self.hearing.forward_propagation(word))
        return (result == action_requested).all()

    def change_score(self, points):
        self.score += points

    def check_sleeping(self):
        if self.sleep >= self.turns_to_sleep:
            # goes to sleep
            self.sleep = 0
            self.now = 1
        else:
            self.sleep += 1
            self.now = 0

    def is_ready(self):
        return self.now == 0

    def is_sleeping(self):
        return self.now == 1

    def is_waiting(self):
        return self.now == 2
