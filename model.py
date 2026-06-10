import numpy as np

def evaluate(muL, nL, pL, number_of_interpolations):
    """
    Evaluate the polynomial

        y = muL + nL*x + pL*x^3

    Parameters
    ----------
    muL : float
        Constant term.
    nL : float
        Linear coefficient.
    pL : float
        Cubic coefficient.
    number_of_interpolations : int
        Number of x points between 0 and 1.

    Returns
    -------
    x : numpy.ndarray
        x values from 0 to 1.
    y : numpy.ndarray
        Corresponding y values.
    """

    x = np.linspace(0.0, 1.0, int(number_of_interpolations))
    y = muL + nL * x + pL * x**3

    return x, y
