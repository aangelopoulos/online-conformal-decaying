import numpy as np
import pdb

# Pinball loss function
def pinball_loss(y, yhat, tau):
    # tau is the quantile
    return np.maximum(tau * (y - yhat), (tau - 1) * (y - yhat))

def get_online_quantile(scores, q_1, etas, alpha):
    """
    Computes the online quantile of a set of scores.
    :param scores: (np.array) The scores.
    :param q_1: (float) The quantile to compute.
    :param eta: (np.array) The sequence of learning rates.
    :return: (float) The sequence of online quantiles.
    """
    T = scores.shape[0]
    q = np.zeros(T)
    q[0] = q_1
    for t in range(T):
        err_t = (scores[t] > q[t]).astype(int)
        if t < T - 1:
            q[t + 1] = q[t] - etas[t] * (alpha - err_t)
    return q

def dtACI(scores, alpha, gammas=[0.001,0.005,0.01,0.05,0.1,0.2,0.5,1]): # param1 is sigma in dtACI work, param2 is eta in dtACI work
    T = scores.shape[0]
    K = len(alphat_inits)
    q = np.zeros(T)
    # Copy the initial alpha values
    alphat_inits = np.ones_like(gammas) * alpha
    alphats = np.copy(alphat_inits)
    # Eta parameter in dtACI
    desired_window_size = 100
    param1 = 1/(2*desired_window_size)
    param2 = np.sqrt((np.log(K*desired_window_size)+2)/((1-alpha)**2*(alpha**2))) * np.sqrt(3/desired_window_size)

    # Initialization
    weights = np.ones(K)  # Initialize weights for each candidate

    # Main loop of the algorithm
    for t in range(1, T + 1):
        score = scores[t]

        # Calculate probabilities based on weights
        probabilities = weights / np.sum(weights)

        # Output alpha_t with the calculated probability
        alpha_t = np.random.choice(alphats, p=probabilities)

        # Update weights based on the pinball loss between the score and alpha[i]
        for i in range(K):
            loss = pinball_loss(score, alpha_t, 1-alpha)
            weights[i] *= np.exp(-param2 * loss)


        # Calculate a new weight sum W_t
        W_t = np.sum(weights)

        # Update weights with a regularization term involving sigma
        weights = (1 - param1) * weights + W_t * param1 / K

        err_t = np.array([score <= np.quantile(scores[:t], 1-alphats[i], method="higher") for i in range(K)])
        q = np.quantile(scores[:t], 1-alpha_t, method="higher")

        # Adjust alpha for the next time step based on the errors
        for i in range(k):
            alphats[i] += gammas[i] * (alpha - err_t[i])

    return q
