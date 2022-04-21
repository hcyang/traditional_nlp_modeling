# -*- coding: utf-8 -*-
# ---- NOTE-OPTIONAL-CODING ----
# -*- coding: latin-1 -*-
"""
This module tests common parallel-programming helper functions defined in parallel_helper.
"""

from utility.parallel_helper.parallel_helper import ParallelHelper

from utility.debugging_helper.debugging_helper import DebuggingHelper

def main():
    """
    The main() to quickly test ParallelHelper functions
    """
    DebuggingHelper.write_line_to_system_console_out(
        f'number of cores = {ParallelHelper.get_number_cores()}')

if __name__ == '__main__':
    main()
