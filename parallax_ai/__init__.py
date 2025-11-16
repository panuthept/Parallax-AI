"""Parallax - A package for parallel multi-agent inference"""

__version__ = "0.5.1"

from .composer import OutputComposer
from .datapool import DataPool
from .distributor import Distributor
from .service import Service
from .benchmarks import SEASafeguardBench, SEALSBench, PKUSafeRLHFQA