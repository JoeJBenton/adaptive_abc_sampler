import numpy as np


class UniformPrior:

    def __init__(self):
        pass

    def sample(self):
        return np.random.uniform(0, 1)

    def density(self, theta):
        if 0 <= theta <= 1:
            return 1
        else:
            return 0


class NormalLikelihood:

    def __init__(self, sample_size):
        self.sample_size = sample_size

    def single_sample(self, theta):
        return np.random.normal(theta, 1)

    def full_sample(self, theta):
        return np.array([self.single_sample(theta) for _ in range(self.sample_size)])


class NormalPerturbationKernel:

    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, particle):
        (theta, X) = particle
        theta = np.random.normal(theta, self.sigma)
        X = [np.random.normal(x, self.sigma) for x in X]
        return theta, X


class EuclideanMetricWithMean:

    def __init__(self):
        pass

    def measure(self, y_1, y_2):
        return np.abs(np.average(y_1) - np.average(y_2))

    def indicator(self, x, y, epsilon):
        if self.measure(x, y) < epsilon:
            return 1
        else:
            return 0