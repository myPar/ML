import math


class Optimizer(object):
    def __init__(self, name):
        self.name = name

    def get_next_coefficient(self, net_parameters):
        pass


class Adam(Optimizer):
    def __init__(self):
        super().__init__("Adam")
        self.betta1 = 0.9
        self.betta2 = 0.99
        self.m_prev = 0
        self.v_prev = 0
        self.step = 1
        self.epsilon = 0.000001

    def get_next_coefficient(self, grad_norm):
        assert grad_norm >= 0

        m_cur = self.m_prev * self.betta1 + (1 - self.betta1) * grad_norm
        v_cur = self.v_prev * self.betta2 + (1 - self.betta2) * (grad_norm ** 2)

        self.step += 1
        self.m_prev = m_cur
        self.v_prev = v_cur

        arg1 = self.m_prev / (1 - self.betta1 ** self.step)
        arg2 = self.v_prev / (1 - self.betta2 ** self.step)

        result = arg1 / math.sqrt(arg2 + self.epsilon)

        return result


def get_norm(get_norm):
    return 1
