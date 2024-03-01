# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# udpated April 20
# specific xhat supplied (copied from xhatlooper_bounder by DLW, Dec 2019)

import mpisppy.cylinders.spoke as spoke
from mpisppy.extensions.xhatspecific import XhatSpecific
from mpisppy.utils.xhat_eval import Xhat_Eval

import mpisppy.MPI as mpi
import logging

fullcomm = mpi.COMM_WORLD
global_rank = fullcomm.Get_rank()
fullcom_n_proc = fullcomm.Get_size()

logger = logging.getLogger(__name__)


############################################################################
class XhatSpecificInnerBound(spoke.InnerBoundNonantSpoke):

    converger_spoke_char = 'S'

    def ib_prep(self):
        """
        Set up the objects needed for bounding.

        Returns:
            xhatter (xhatspecific object): Constructed by a call to Prep
        """
        if "bundles_per_rank" in self.opt.options\
           and self.opt.options["bundles_per_rank"] != 0:
            raise RuntimeError("xhat spokes cannot have bundles (yet)")

        if not isinstance(self.opt, Xhat_Eval):
            raise RuntimeError("XhatShuffleInnerBound must be used with Xhat_Eval.")

        verbose = self.opt.options['verbose']
        xhatter = XhatSpecific(self.opt)
        # somehow deal with the prox option .... TBD .... important for aph APH

        # begin iter0 stuff
        xhatter.pre_iter0()
        self.opt._save_original_nonants()

        self.opt._lazy_create_solvers()  # no iter0 loop, but we need the solvers

        self.opt._update_E1()  
        if (abs(1 - self.opt.E1) > self.opt.E1_tolerance):
            if self.opt.cylinder_rank == 0:
                logger.error("ERROR")
                logger.error(f"Total probability of scenarios was {self.opt.E1}")
                logger.error(f"E1_tolerance = {self.opt.E1_tolerance}")
            quit()

        ### end iter0 stuff

        xhatter.post_iter0()
        self.opt._save_nonants()  # make the cache

        return xhatter

    def main(self):
        """
        Entry point. Communicates with the optimization companion.

        """
        verbose = self.opt.options["verbose"] # typing aid  
        logger.debug(f"Enter xhatspecific main on rank {global_rank}")

        # What to try does not change, but the data in the scenarios should
        xhat_scenario_dict = self.opt.options["xhat_specific_options"]\
                                             ["xhat_scenario_dict"]

        xhatter = self.ib_prep()

        ib_iter = 1  # ib is for inner bound
        got_kill_signal = False
        while not self.got_kill_signal():
            if (ib_iter - 1) % 10000 == 0:
                logger.debug(f"   IB loop iter={ib_iter} on {global_rank=}")
                logger.debug(f"   IB got from opt on {global_rank=}")
            if self.new_nonants:
                logger.debug(f"  and its new! on {global_rank=}")
                logger.debug(f"  localnonants={self.localnonants}")

                self.opt._put_nonant_cache(self.localnonants)  # don't really need all caches
                self.opt._restore_nonants()
                innerbound = xhatter.xhat_tryit(xhat_scenario_dict, restore_nonants=False)

                self.update_if_improving(innerbound)

            ib_iter += 1

        logger.debug(f'IB specific thread ran {ib_iter} iterations')
    
if __name__ == "__main__":
    print("no main.")
