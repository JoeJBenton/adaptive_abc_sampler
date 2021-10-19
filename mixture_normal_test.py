import numpy as np


class UniformPrior:

    def __init__(self):
        pass

    def sample(self):
        return np.random.uniform(-10, 10)

    def density(self, theta):
        if -10 <= theta <= 10:
            return 0.05
        else:
            return 0


class MixedNormalLikelihood:

    def __init__(self):
        pass

    def single_sample(self, theta):
        if np.random.rand() < 0.5:
            return np.random.normal(theta, 1)
        else:
            return np.random.normal(theta, 0.01)

    def full_sample(self, theta):
        return self.single_sample(theta)


class NormalMHKernel:

    def __init__(self, likelihood, prior, metric):
        self.likelihood = likelihood
        self.prior = prior
        self.metric = metric

    def theta_proposal(self, theta, sigma):
        return np.random.normal(theta, sigma)

    # def theta_proposal_density(self, theta_0, theta_1, sigma):
    #     return np.exp(-(theta_1 - theta_0)*(theta_1 - theta_0)/(2*sigma*sigma))  # Include normalisation constant

    def sample(self, particle, sigma, y_obs, epsilon):
        (theta, X) = particle

        new_theta = self.theta_proposal(theta, sigma)
        new_X = [self.likelihood.full_sample(theta) for _ in range(len(X))]

        indicator_numerator = sum([self.metric.indicator(x, y_obs, epsilon) for x in new_X])
        indicator_denominator = sum([self.metric.indicator(x, y_obs, epsilon) for x in X])
        indicator_ratio = indicator_numerator / indicator_denominator
        # Don't need proposal ratio because transition density is symmetric
        prior_ratio = self.prior.density(new_theta) / self.prior.density(theta)
        acceptance_prob = min(1, indicator_ratio * prior_ratio)

        if np.random.rand() < acceptance_prob:
            return new_theta, new_X
        else:
            return theta, X


class EuclideanMetric:

    def __init__(self):
        pass

    def measure(self, y_1, y_2):
        return np.abs(y_1 - y_2)

    def indicator(self, x, y, epsilon):
        if self.measure(x, y) < epsilon:
            return 1
        else:
            return 0
