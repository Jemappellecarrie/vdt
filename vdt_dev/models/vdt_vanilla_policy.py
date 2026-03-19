from __future__ import annotations

from vdt_dev.models.vdt_bar_policy import VDTBARPolicy


class VDTVanillaPolicy(VDTBARPolicy):
    """
    Same-stack dev baseline using the Step 1/2/3 policy shell without BAR routing.

    This keeps tokenization, runner plumbing, checkpoints, logging, and evaluation
    identical to the dev stack while using the transformer's standard residual path.
    """

    def __init__(self, *args, **kwargs):
        kwargs["use_attnres"] = False
        super().__init__(*args, **kwargs)
