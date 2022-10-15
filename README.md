# Univariate_Gaussian_MLE

## Problem Statement

I have $n=100$ pieces of indepdendent and identicially distrubuted (i.i.d) data related to some measurement that is drawn from a **normal distribution**.

This code computes $\mu_{MLE}$ and $\sigma_{MLE}$, which represent the mean and standard deviation computed via MLE, respectively, of the underlying normal distribution for the dataset "MLE_dataset.npy"

## Variable Definitions

1. x = collection of data points ($x_1,x_2,...,x_n$). Loaded in as a numpy array of shape (100,)
2. $\mu$ = mean of the normal distribution
3. $\sigma$ = standard deviation of the normal distribution

## Useful Numpy functions

1. x.shape : returns a tuple that contains the dimension(s) of the numpy array variable name "x"

## Starting information

1. You are given "MLE_dataset.npy" which contains the aforementioned data in the form of a numpy array.

2. You are also given a "load" function which, given the path to "MLE_dataset.npy", will load the values into a numpy array variable and return said variable.

3. For a single piece of data, $x_i$, if the data is drawn from a normal distribution, the likelihood of that data given a normal distribution with parameterized by $\mu$ and $\sigma^2$ can be represented as:

$$p(x_i | \mu, \sigma^2) = \mathcal{N}(x_i; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} * \text{exp} \ [-\frac{1}{2}(\frac{x_i - \mu}{\sigma})^2]$$

4. In our case, we aren't dealing with a single observation, but $n=100$ observations. Because our observations are **independent**, we can represent our likelihood function for all observations as the product of the individual observations:

$$p(x | \mu, \sigma^2) = \Pi_{i=1}^N \mathcal{N}(x_i; \mu, \sigma^2) = (\frac{1}{2\pi\sigma^2})^{n/2} * \text{exp} \ [-\frac{1}{2}\sum_{i=1}^N (\frac{x_i - \mu}{\sigma})^2]$$

## MLE estimates for $\mu$ and $\sigma^2$ 

$$\mu_{MLE} = \frac{1}{n} \sum_{i=1}^n x_i$$
$$ \sigma^2_{MLE} = \frac{1}{n} \sum_{i=1}^n(x_i-\mu_{MLE})^2$$


## Task1

1. Write code that correctly finds $\mu_{MLE}$ and $\sigma_{MLE}$ for the data set "MLE_dataset.npy"


Let us add a layer of difficulty to the problem of estimating the underlying distribution(s) of our data. Previously. we knew that all of our data came from a single underlying normal distribution.

For this problem, I have gone out and collected a brand new data set consisting of $n=200$ data points. These data points are drawn from **one of two unknown gaussian distributions**.

To keep the notation consistent, I will use the subscript ID $k = \{0,1\}$ to represent which gaussian a parameter/variable is referring to.

For instance, $\mu_k$ represents the mean of the gaussian with ID $k$. $\mu_0$ represents the mean of the first gaussian, and $\mu_1$ represents the mean of the second gaussian

## Variable definitions

1. $k$: id of the gaussian distributions. {0,1} 
2. $n$: number of data points. 200 in this case
3. $\mu_k$: the mean of normal distribution $k$
4. $\sigma_k$: the std of normal distribution $k$
5. $\pi_k$: the prior probability of normal distribution $k$
7. $t$: current step number
8. $z_i = k$: represents the idea that i-th data point was drawn from gaussian $k$
9. $x_i$: i-th data point



To make things clear, here is a list of things we do and don't know:

**known:**
1. Our data points, $X = x_1, x_2, ..., x_n$ for $n = 1$ to $n=200$
2. $\mu_0^{t=0}, \sigma_0^{t=0}, \pi_0^{t=0}$ which represent an initial guess for the  mean, std, and prior  of the first gaussian with ID $k=0$
3. $\mu_1^{t=0}, \sigma_1^{t=0}, \pi_1^{t=0}$ which represent an initial guess for the  mean, std, and prior  of the second gaussian with ID $k=1$
4. $\pi_0^{t=0}$ and $\pi_1^{t=0}$ which represent an initial guess for the prior probabilities of gaussian 0 and gaussian 1, respectively
5. The number of iterations your EM algorithm should run for. Do not worry about early stopping, just let it run the full number of iterations
6. Let us define the variable $\theta_k^t$ to be the set ($\hat{\mu_k^t}, \hat{\sigma_k^t}, \hat{\pi_k^t})$, which is our current estimate at time step t for the three unknown parameters of gaussian k. 

**not-know:**
1. The true means, $\mu_0$ and $\mu_1$ for both gaussian distributions
2. The true standard deviations, $\sigma_0$ and $\sigma_1$ for both gaussian distributions 
3. Which gaussian distribution, $k=0$ or $k=1$, a point, $x_i$, came from.
4. The true fraction of our points that came from gaussian $k=0$ and the true fraction of our points that came from gaussian $k=1$ (the prior probabilities)


End goal is to implement an algorithm that implements EM and returns the following:
1. $\hat{\mu_0}, \hat{\sigma_0}, \hat{\pi_0}$ which represent the final EM estimate the mean, standard deviation (std), and prior of the first gaussian with ID $k=0$
2. $\hat{\mu_0}, \hat{\sigma_0}, \hat{\pi_0}$ which represent the final EM estimate of the mean, standard deviation (std), and prior of the first gaussian with ID $k=1$


### Useful functions

1. array.shape - returns the shape as a tuple of a numpy array
2. scipy.stats.norm(mean, std).pdf(value) - returns the value pdf(val) of a gaussian distribution paramaterized by (mean, std)


### Task Order
1. Determine the update rules for $\mu_k^{t+1}$ and $\sigma_k^{t+1}$ and write them in the cell below
2. Code the E and M step
3. Run and test your code on "EM_dataset.py"


## E - Step
$P(z_i = k | x_i, \theta_k^t)$ reflects the responsibility the k-th gaussian has for the i-th data point
$$P(z_i = k | x_i, \theta_k^t) = \frac{P(x_i | z_i = k, \theta_k^t)*\pi_k^t}{P(x_i | \theta_k^t)}$$ \




## M - Step


$$\mu_k^{t+1} = \ ??$$

$$\sigma_k^{t+1} = \ ??$$

$$\pi_k^{t+1} = \frac{\sum_{i=1}^n P(z_i = k | x_i, \theta_k^t)}{n}$$
3. Which gaussian distribution, $k=0$ or $k=1$, a point, $x_i$, came from.
