# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module implements base data manager objects
"""

class BaseRecordGenerator:
    """
    Base class for record generator objects.
    """
    # ---- NOTE-PYLINT ---- R0903: Too few public methods
    # pylint: disable=R0903
    # ---- NOTE-PYLINT ---- R0902: Too many instance attributes
    # pylint: disable=R0902
    def __init__(self):
        pass

    def generate(self):
        # ---- NOTE-ABSTRACT-FUNCTION ----
        """
        Generator method.
        """
        raise NotImplementedError()
