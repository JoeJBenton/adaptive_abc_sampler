import numpy as np
import matplotlib.pyplot as plt


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

    def sample(self, theta_0):
        return np.random.normal(theta_0, self.sigma)

    def density(self, theta_0, theta_1):
        return np.exp(-((theta_0 - theta_1)**2)/(2*(self.sigma**2)))


def euclidean_metric_with_mean(y_1, y_2):
    return np.abs(np.average(y_1) - np.average(y_2))


def ABC_SMC_sampler(y_obs, likelihood, prior, metric, epsilon_sequence, kernel):

    particle_count = 1000
    epsilon = epsilon_sequence[0]

    particles = []
    trial_count = 0
    for i in range(particle_count):
        sample = None
        while not sample:
            theta = prior.sample()
            y_synth = likelihood.full_sample(theta)

            if metric(y_obs, y_synth) < epsilon:
                sample = theta

            trial_count += 1

        particles.append(sample)

    weights = [float(1) / particle_count for _ in range(particle_count)]

    print("\nTime: 0")
    print("Acceptance rate: %s" % (float(particle_count) / trial_count))
    print("Sample mean: %s" % np.average(particles))
    print("Variance: %s" % np.var(particles))

    for t in range(1, len(epsilon_sequence)):
        epsilon = epsilon_sequence[t]
        new_particles = []
        new_weights = []

        trial_count = 0

        while len(new_particles) < particle_count:
            theta = np.random.choice(particles, p=weights)
            theta = kernel.sample(theta)

            if prior.density(theta) > 0:
                y_synth = likelihood.full_sample(theta)
                if metric(y_obs, y_synth) < epsilon:
                    new_particles.append(theta)

                    normalisation = sum([weights[j] * kernel.density(theta, particle)
                                         for (j, particle) in zip(range(particle_count), particles)])
                    new_weights.append(prior.density(theta) / normalisation)

            trial_count += 1

        weight_sum = sum(new_weights)
        new_weights = new_weights / weight_sum

        particles = new_particles
        weights = new_weights

        print("\nTime: %s" % t)
        print("Acceptance rate: %s" % (float(particle_count) / trial_count))
        print("Sample mean: %s" % np.average(particles))
        print("Variance: %s" % np.var(particles))

    return particles


if __name__ == "__main__":
    my_prior = UniformPrior()
    my_likelihood = NormalLikelihood(100)
    my_kernel = NormalPerturbationKernel(0.01)

    true_theta = my_prior.sample()
    print("Theta: %s" % true_theta)

    observations = my_likelihood.full_sample(true_theta)
    epsilon_schedule = [0.1, 0.04, 0.02, 0.01, 0.004, 0.002, 0.001]

    print("Observed mean: %s" % np.average(observations))

    approx_posterior_sample = ABC_SMC_sampler(observations, my_likelihood, my_prior, euclidean_metric_with_mean,
                                              epsilon_schedule, my_kernel)

    plt.hist(approx_posterior_sample)
    plt.show()
