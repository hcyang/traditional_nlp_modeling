# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module provides some common parallel-programming helper functions.
"""

import multiprocessing

from joblib import Parallel, delayed

class ParallelHelper:
    """
    This class contains some common functions for parallel programming.
    """

    @staticmethod
    def get_parallelism(number_cores_deduction: int = 1) -> int:
        """
        Return parallelism.
        """
        parallelism: int = ParallelHelper.get_number_cores()
        parallelism -= number_cores_deduction
        if parallelism <= 0:
            parallelism = 1
        return parallelism

    @staticmethod
    def get_active_children() -> int:
        """
        Return the number of active children.
        """
        number_cores: int = multiprocessing.active_children()
        return number_cores

    @staticmethod
    def get_number_cores() -> int:
        """
        Return the number of cores of this machine.
        """
        number_cores: int = multiprocessing.cpu_count()
        return number_cores

    @staticmethod
    def square(a_list, i):
        """
        Compute the square of an item in a_list.
        """
        return a_list[i] * a_list[i]

    @staticmethod
    def compare(a_list, i) -> int:
        """
        Test if numbers in a_list is in order.
        """
        return 0 if a_list[i] < a_list[i+1] else 1

    @staticmethod
    def example_function_joblib(number: int = 1000):
        """
        Test joblib behavior.
        Example snippet to run this test function:
            from utility.parallel_helper.parallel_helper import ParallelHelper
            ParallelHelper.test_joblib(1000)
        """
        # ---- NOTE-PYLINT ---- C0301: Line too long
        # pylint: disable=C0301
        n_jobs = 8
        n_range = range(number)
        a_list = [x for x in n_range]
        # ---- NOTE: the first implementation is errorneous, c_sum != 0 ---- i in lambda function were indeterminate!
        # ---- NOTE: the code will cause a warning: W0640: Cell variable i defined in loop (cell-var-from-loop)
        # ==== b_list = Parallel(n_jobs=n_jobs)(delayed(lambda x: x[i] * x[i])(a_list) for i in range(len(a_list)))
        # ==== c_list = Parallel(n_jobs=n_jobs)(delayed(lambda x: 0 if x[i] < x[i+1] else 1)(b_list) for i in range(len(b_list)-1))
        # ==== c_sum = sum(c_list)
        # ==== print(f'c_sum={c_sum}')
        # ---- NOTE: the third implementation is OK, c_sum == 0
        b_list = Parallel(n_jobs=n_jobs)(delayed(lambda x, i: x[i] * x[i])(a_list, i) for i in range(len(a_list)))
        c_list = Parallel(n_jobs=n_jobs)(delayed(lambda x, i: 0 if x[i] < x[i+1] else 1)(b_list, i) for i in range(len(b_list)-1))
        c_sum = sum(c_list)
        print(f'c_sum={c_sum}')
        # ---- NOTE: the third implementation is OK, c_sum == 0
        b_list = Parallel(n_jobs=n_jobs)(delayed(ParallelHelper.square)(a_list, i) for i in range(len(a_list)))
        c_list = Parallel(n_jobs=n_jobs)(delayed(ParallelHelper.compare)(b_list, i) for i in range(len(b_list)-1))
        c_sum = sum(c_list)
        print(f'c_sum={c_sum}')
