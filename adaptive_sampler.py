import matplotlib.pyplot as plt
from mixture_normal_test import *


def ESS(weights):
    if not weights:
        # To deal with the case of degenerate weights
        return 1
    else:
        return 1.0/(np.linalg.norm(weights)**2)


def updated_weights(y_obs, particles, old_weights, old_epsilon, new_epsilon, metric):
    new_weights = []
    for (particle, weight) in zip(particles, old_weights):
        (theta, X) = particle
        # print("X", X)
        # print("y_obs", y_obs)
        # input("...")
        numerator = sum([metric.indicator(x, y_obs, new_epsilon) for x in X])
        denominator = sum([metric.indicator(x, y_obs, old_epsilon) for x in X])

        # Deal with totally degenerate weights
        if denominator == 0:
            print("Degenerate weights...")
            return False

        new_weights.append(weight * numerator / denominator)
    normalisation = sum(new_weights)
    new_weights = [(w / normalisation) for w in new_weights]
    return new_weights


def find_new_epsilon(y_obs, particles, old_weights, alpha, old_epsilon, min_epsilon, metric):
    target = alpha*ESS(old_weights)
    # print("Target ESS: %s" % target)

    candidate_ESS = ESS(updated_weights(y_obs, particles, old_weights, old_epsilon, min_epsilon, metric))
    # print("Min epsilon ESS: %s" % candidate_ESS)
    if candidate_ESS > target:
        return min_epsilon

    epsilon_upper = old_epsilon
    candidate_ESS_upper = ESS(old_weights)
    epsilon_lower = min_epsilon
    candidate_ESS_lower = candidate_ESS

    tolerance = 0.01
    while candidate_ESS_upper - candidate_ESS_lower > tolerance*target:
        # print("Upper epsilon: %s, Upper ESS: %s" % (epsilon_upper, candidate_ESS_upper))
        # print("Lower epsilon: %s, Lower ESS: %s" % (epsilon_lower, candidate_ESS_lower))
        epsilon_bisect = (epsilon_upper + epsilon_lower) / 2
        candidate_ESS = ESS(updated_weights(y_obs, particles, old_weights, old_epsilon, epsilon_bisect, metric))
        if candidate_ESS > target:
            epsilon_upper = epsilon_bisect
            candidate_ESS_upper = candidate_ESS
        else:
            epsilon_lower = epsilon_bisect
            candidate_ESS_lower = candidate_ESS

    return (epsilon_upper + epsilon_lower) / 2


def adaptive_sampler(y_obs, likelihood, prior, kernel, metric):

    iteration = 0
    num_of_particles = 10000
    resample_threshold = 5000
    particles = []
    weights = []
    M = 10
    epsilon = 100
    min_epsilon = 0.01
    alpha = 0.95

    for _ in range(num_of_particles):
        theta = prior.sample()
        X = [likelihood.full_sample(theta) for _ in range(M)]
        particles.append((theta, X))
        weights.append(1.0 / num_of_particles)

    while epsilon > min_epsilon:
        # Logging data
        iteration += 1
        print("Iteration: %s" % iteration)
        print("Number of alive particles: %s" % len(particles))

        # Bisection algorithm to calculate new_epsilon
        new_epsilon = find_new_epsilon(y_obs, particles, weights, alpha, epsilon, min_epsilon, metric)
        print("Epsilon: %s" % new_epsilon)

        # Calculate and normalise weights
        new_weights = []
        for (particle, weight) in zip(particles, weights):
            (theta, X) = particle
            numerator = sum([metric.indicator(x, y_obs, new_epsilon) for x in X])
            denominator = sum([metric.indicator(x, y_obs, epsilon) for x in X])
            new_weights.append(weight * numerator / denominator)
        normalisation = sum(new_weights)
        weights = [(w / normalisation) for w in new_weights]

        epsilon = new_epsilon

        # Resampling step
        if ESS(weights) < resample_threshold:
            print("Resampling...")
            new_particle_indices = [np.random.choice(np.arange(len(particles)), p=weights)
                                    for _ in range(num_of_particles)]
            particles = [particles[i] for i in new_particle_indices]
            weights = [1.0/num_of_particles for _ in range(num_of_particles)]

        # Perturb particles
        # TODO Pick a sensible normalisation here
        # TODO And pick a more sensible form of kernel
        new_particles = []
        new_weights = []
        for (particle, weight) in zip(particles, weights):
            if weight > 0:
                new_particles.append(kernel.sample(particle, 0.01, y_obs, epsilon))
                new_weights.append(weight)
        particles = new_particles
        weights = new_weights

        # Logging code
        thetas = [theta for (theta, _) in particles]
        print("Sample average: %s" % np.average(thetas))
        print("Sample variance: %s" % np.var(thetas))

        if iteration % 30 == 0:
            hist_bins = np.linspace(-5, 5, 101)
            plt.hist(thetas, bins=hist_bins, density=True, weights=weights)
            plt.show()

    return particles, weights


def main():
    prior = UniformPrior()
    likelihood = MixedNormalLikelihood()
    metric = EuclideanMetric()
    kernel = NormalMHKernel(likelihood, prior, metric)

    true_theta = 0
    true_obs = likelihood.full_sample(true_theta)
    print("True theta: %s" % true_theta)
    print("Observed mean: %s" % np.average(true_obs))

    (particles, weights) = adaptive_sampler(true_obs, likelihood, prior, kernel, metric)

    thetas = [theta for (theta, _) in particles]
    plt.hist(thetas, density=True, weights=weights)
    plt.show()


if __name__ == "__main__":
    main()
