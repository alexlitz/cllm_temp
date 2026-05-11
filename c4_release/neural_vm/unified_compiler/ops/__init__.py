"""Per-layer / per-concern op factory modules.

Splits the legacy 4800-line ``migrated_ops.py`` into navigable per-layer
files (L0 .. L16), plus dedicated modules for ALU composites, flag-gated
factories, and model-level post-passes. The public API is preserved via
star-imports here; the legacy ``migrated_ops.py`` is now a thin re-export
shim over this package.
"""

from .shared import *  # noqa: F401,F403
from .l0_ops import *  # noqa: F401,F403
from .l1_ops import *  # noqa: F401,F403
from .l2_ops import *  # noqa: F401,F403
from .l3_ops import *  # noqa: F401,F403
from .l4_ops import *  # noqa: F401,F403
from .l5_ops import *  # noqa: F401,F403
from .l6_ops import *  # noqa: F401,F403
from .l7_ops import *  # noqa: F401,F403
from .l8_ops import *  # noqa: F401,F403
from .l9_ops import *  # noqa: F401,F403
from .l10_ops import *  # noqa: F401,F403
from .l11_ops import *  # noqa: F401,F403
from .l12_ops import *  # noqa: F401,F403
from .l13_ops import *  # noqa: F401,F403
from .l14_ops import *  # noqa: F401,F403
from .l15_ops import *  # noqa: F401,F403
from .l16_ops import *  # noqa: F401,F403
from .alu_ops import *  # noqa: F401,F403
from .flag_gated_ops import *  # noqa: F401,F403
from .model_ops import *  # noqa: F401,F403
from .all_core_ops import all_core_ops, all_alu_postop_attach_ops  # noqa: F401
