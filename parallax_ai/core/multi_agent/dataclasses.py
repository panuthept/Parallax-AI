from uuid import UUID, uuid4
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any


@dataclass
class Dependency:
    external_data: Optional[List[str]] = None
    agent_outputs: Optional[List[str]] = None


@dataclass
class AgentIO:
    dependency: Optional[Dependency] = None
    input_processing: Optional[Callable[[list, dict], list]] = None # (outputs, data) -> inputs
    output_processing: Callable[[list, list, dict], list] = None # (inputs, outputs, data) -> processed_outputs


@dataclass
class Package:
    id: UUID = field(default_factory=lambda: uuid4())
    agent_inputs: Dict[str, Any] = field(default_factory=dict)    # agent_name -> inputs
    agent_outputs: Dict[str, Any] = field(default_factory=dict)   # agent_name -> outputs
    external_data: Dict[str, Any] = field(default_factory=dict)      # data_name -> data_value
