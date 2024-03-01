# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.

# Register numpy types in Pyomo, see https://github.com/Pyomo/pyomo/issues/3091
from pyomo.common.dependencies import numpy_available as _np_avail
bool(_np_avail)

from mpisppy.MPI import COMM_WORLD, _haveMPI as haveMPI
from mpisppy.log import global_toc

_global_rank = COMM_WORLD.rank
