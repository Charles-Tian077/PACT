"""Independent, untrained geometry component; no risk certificate is implied."""
from .core import PairHead, ProposalHead, AcceptedUpdateBudget, paired_features
from .runtime import generate_with_recourse
__all__ = ['PairHead','ProposalHead','AcceptedUpdateBudget','paired_features','generate_with_recourse']
