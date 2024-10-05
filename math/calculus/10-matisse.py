#!/usr/bin/env python3
'''
        Calculate the derivative of a polynomial.
'''


def poly_derivative(poly):
    '''
        a function def poly_derivative(poly):
    that calculates the derivative of a polynomial
    '''
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
