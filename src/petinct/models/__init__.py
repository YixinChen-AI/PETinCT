"""PETinCT model definitions."""

from petinct.models.organ_relation import (
    WholeBodyOrganMetabolicRelationNet,
)
from petinct.models.ct_conditioned_organ_relation import (
    CtConditionedWOMRNet,
)

__all__ = ["WholeBodyOrganMetabolicRelationNet"]
