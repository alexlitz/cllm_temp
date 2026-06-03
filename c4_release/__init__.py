"""c4_release package marker.

Lets tests reach modules via the ``c4_release.neural_vm.*`` namespace as
well as the bare ``neural_vm.*`` one used elsewhere in the suite (the
latter still works when the inner ``c4_release/`` dir is on ``sys.path``).
"""
