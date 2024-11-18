import sympy as sp

def newton_cotes_h2(f, a, b):
    """
    Implements Newton-Cotes order h2 for numerical integration.

    Parameters:
        f (function): The function to integrate.
        a (float): The lower limit of integration.
        b (float): The upper limit of integration.

    Returns:
        float: Approximation of the integral.
    """

    # Define the variable and function
    x = sp.symbols('x')
    
    # Differentiate the function twice
    f_prime = sp.diff(f, x, 1)  # First derivative
    f_double_prime = sp.diff(f, x, 2)  # Second derivative
    
    # Evaluate the second derivative at the limits a and b
    f_double_prime_a = abs(f_double_prime.subs(x, a))
    f_double_prime_b = abs(f_double_prime.subs(x, b))
    
    # Maximum of the second derivative at a and b
    max_f_double_prime = max(f_double_prime_a, f_double_prime_b)
    
    # Calculate the appropriate step size h using the error formula
    h = sp.sqrt((12 * 0.5 * 10**-2 * (b - a)) / max_f_double_prime)
    
    # Choose an appropriate value for h (just using the computed value here)
    h_value = h.evalf()
    
    # Midpoint for Simpson's Rule
    m = (a + b) / 2
    
    # Calculate the integral using the Newton-Cotes (Simpson's Rule)
    integral = (h_value / 2) * (f.subs(x, a) + f.subs(x, b) + 2 * f.subs(x, m))
    
    return integral, h_value

