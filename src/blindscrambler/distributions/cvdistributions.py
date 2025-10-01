# Metadata:
# Author: Syed Raza
# email: sar0033@uah.edu

# use python secrets to make a normally distributed random samples
import secrets
import numpy as np


def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """
    Cryptographically secure sample
    """
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53) # in [0, 1)

    return a + (b - a) * u

def exponentialdist(lam):
    """
    Params: 
    (1) lam : double
        The number lambda that will be used to generate samples of the exponential distributions
    
    Returns:
    (2) x : double
        A random sample that is exponentially distributed

    Function description: Basically, this function performs Inverse Transform Sampling. You take a random number generated between [0, 1]
    This represets the CDF of some given distribution. Then you apply the inverse CDF function to generate a sample. The given sample will
    be from the distribution in question.

    NOTE: the inverse CDF used here is from the exponential distribution. Therefore, the generated numbers will be exponentially distributed
    """

    # use the unifrom function above to generate a uniformly distributed sample in the range --> [0, 1]
    y = uniform()

    # use this y to generate the exponential sample
    x = -(1/lam) * np.log(y)

    # return this sample :)
    return x

def poissondist(lam):
    """
    Params: 
    (1) lam : double
        The number lambda that will be used to generate samples of the poisson distributions
    
    Returns:
    (2) x : double
        A random sample that is poisson distributed
    """
        # make cumulative function and k 
    cum = 0
    x = 0

    # make a random sample between (0, 1)
    u = uniform()

    # start with first probability mass 
    p = np.exp(-lam)
    cum += p

    # keep adding probabilities unyil condition is met
    while cum < u:
        x += 1
        p *= lam / x
        cum += p

    return x

if __name__ == "__main__":
    print("testing")

