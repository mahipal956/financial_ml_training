"""
Log-Uniform Parameter Distribution Module (AFML Snippet 9.4)
Enables non-linear parameter sampling across exponential ranges for SVM and regularization parameters.
"""
import numpy as np
from scipy.stats import rv_continuous


class logUniform_gen(rv_continuous):
    """
    Random variable generating log-uniformly distributed numbers between a and b.
    log(x) ~ Uniform[log(a), log(b)]
    """
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def logUniform(a=1e-3, b=1e3):
    """
    Constructs a log-uniform distribution instance.
    """
    return logUniform_gen(a=a, b=b, name="logUniform")
