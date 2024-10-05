#!/usr/bin/env python3
'''
        Calculate the derivative of a polynomial.

    Args:
    poly (list): A list of coefficients representing a polynomial, where the index of each element
                 represents the power of x for that term. For example, if poly = [5, 3, 0, 1],
                 this represents the polynomial f(x) = x^3 + 3x + 5.

    Returns:
    list: A list of coefficients representing the derivative of the polynomial.
          If the derivative is 0, the function returns [0].
          If the input is invalid (not a list or contains non-numeric values), return None.

    Example:
    poly = [5, 3, 0, 1]  # Represents f(x) = x^3 + 3x + 5
    poly_derivative(poly)  # Returns [3, 0, 3] which represents f'(x) = 3x^2 + 3
'''

def poly_derivative(poly):
    # Check if poly is a valid list of integers or floats
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    # Special case for constant polynomial (derivative of a constant is 0)
    if len(poly) == 1:
        return [0]

    # Calculate the derivative: multiply each coefficient by its power (index)
    derivative = [i * poly[i] for i in range(1, len(poly))]

    # If the derivative is empty (all terms canceled out), return [0]
    if len(derivative) == 0:
        return [0]

    return derivative
