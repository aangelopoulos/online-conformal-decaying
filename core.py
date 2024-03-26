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

def get_online_quantile_adaptive(scores, q_1, alpha, decay_rate=0.51, window_size=10, error_threshold_lower=0, error_threshold_upper=1):
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
    etas = scores[0]*np.ones_like(scores)/(np.arange(1, len(scores)+1)**decay_rate)
    errs = np.zeros_like(scores)
    counter = window_size
    for t in range(T):
        errs[t] = (scores[t] > q[t]).astype(float)
        if (t==window_size+1) |((counter <= 0) & ((t - max(t-window_size, 0)) == window_size) & ((errs[t-window_size+1:t+1].mean() >= error_threshold_upper) | (errs[t-window_size+1:t+1].mean() <= error_threshold_lower))):
            print(f"Restarting at time {t}, with local coverage {errs[t-window_size+1:t+1].mean()}")
            # Restart etas.
            Bhat = np.abs(scores[t-window_size+1:t+1]-q[t-window_size+1:t+1]).max()
            etas[t:] = Bhat*np.ones_like(scores[t:])/(np.arange(1, len(scores[t:])+1)**decay_rate)
            counter = window_size
        else:
            counter -= 1
        if t < T - 1:
            q[t + 1] = q[t] - etas[t] * (alpha - errs[t])
    return q

def dtACI(scores, alpha, gammas=[0.0001,0.001,0.05,0.01,0.1,0.2]): # param1 is sigma in dtACI work, param2 is eta in dtACI work
    T = scores.shape[0]
    K = len(gammas)
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
    for t in range(1, T):
        ### Prediction steps
        # Calculate probabilities based on weights
        probabilities = weights / np.sum(weights)

        # Output alpha_t with the calculated probability
        alpha_t = np.random.choice(alphats, p=probabilities)

        # Form the prediction set based on the alpha_t
        q[t] = np.quantile(scores[:t], np.clip(1-alpha_t,0,1), method="higher")

        ### Update steps
        if t < T:
            # Check coverage
            score = scores[t]
            err_t = score > q[t]

            # Update weights based on the pinball loss between the score and alpha[i]
            beta_t = ((scores[:t] > score).sum() / t)
            for i in range(K):
                loss = pinball_loss(np.clip(beta_t,0,1), alpha_t, 1-alpha)
                weights[i] *= np.exp(-param2 * loss)


            # Calculate a new weight sum W_t
            W_t = np.sum(weights)

            # Update weights with a regularization term involving sigma
            weights = (1 - param1) * weights + W_t * param1 / K
            if weights.max() < 1e-10:
                weights /= weights.max()

            q_hypotheticals = [np.quantile(scores[:t], np.clip(1-alphats[i],0,1), method="higher") for i in range(K)]

            err_t_hypotheticals = np.array([score > q_hypotheticals[i] for i in range(K)])

            # Adjust alpha for the next time step based on the errors
            for i in range(K):
                alphats[i] += gammas[i] * (alpha - err_t_hypotheticals[i])

    return q
