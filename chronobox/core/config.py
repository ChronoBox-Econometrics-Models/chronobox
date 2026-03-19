"""Global configuration for chronobox."""

from dataclasses import dataclass


@dataclass
class ChronoBoxConfig:
    """Global configuration."""

    default_optimizer: str = "L-BFGS-B"
    max_iterations: int = 500
    tolerance: float = 1e-8
    default_information_criterion: str = "aicc"
    default_trend: str = "n"
    default_alpha: float = 0.05


# Singleton global config
config = ChronoBoxConfig()
