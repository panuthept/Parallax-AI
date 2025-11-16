from uuid import uuid4
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class Instance:
    id: str = field(default_factory=lambda: uuid4().hex)
    contents: Dict[str, Any] = field(default_factory=dict)  # initial contents
    metadata: Dict[str, Any] = field(default_factory=dict)  # additional metadata
    executors: Dict[str, Any] = field(default_factory=dict) # executor-specific data