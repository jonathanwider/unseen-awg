from dataclasses import dataclass

import numpy as np


# This class currently is only used when trajectories are getting initialized.
@dataclass
class InitTimeLeadTimeMemberState:
    init_time: np.datetime64
    lead_time: np.timedelta64
    ensemble_member: int
