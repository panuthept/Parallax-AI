"""Parallax - A package for parallel multi-agent inference"""

__version__ = "0.5.1"

from .core import (
    DataPool,
    Distributor,
    Service,
    OutputComposer,
    Module,
    ModuleInterface,
    LambdaModule,
    SwitchModule,
    AgentModule,
    AgentSpec,
    ModelSpec,
    AgenticClassificationModule,
    ClassificationModule,
)