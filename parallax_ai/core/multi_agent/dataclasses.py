import yaml
import dill
import types
import inspect
from uuid import uuid4
from copy import deepcopy
from ..agent import Agent
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Union, Optional, Any


@dataclass
class ModuleIO:
    dependencies: List[str]
    input_processing: Optional[Callable[[list, dict], list]] = None # (outputs, data) -> inputs
    output_processing: Callable[[list, list, dict], list] = None # (inputs, outputs, data) -> processed_outputs
    
    def is_lambda_function(self, fn):
        """Check if function is a lambda function."""
        return isinstance(fn, types.LambdaType) and fn.__name__ == "<lambda>"
    
    def _serialize_function(self, func, field_name):
        """Serialize a function (lambda or regular) to a dictionary."""
        if func is None:
            return None
            
        try:
            if self.is_lambda_function(func):
                source = dill.source.getsource(func)
                # Clean up the source by removing field assignment and trailing commas
                source = source.replace(f"{field_name}=", "").strip()
                if source.endswith(','):
                    source = source[:-1].strip()
                return {
                    'type': 'lambda',
                    'source': source
                }
            else:
                return {
                    'type': 'function',
                    'name': func.__name__,
                    'source': inspect.getsource(func).strip()
                }
        except Exception as e:
            print(f"Warning: Could not serialize {field_name} function: {e}")
            return None
    
    def _setup_yaml_multiline_representer(self):
        """Configure YAML to use literal style for multiline strings."""
        yaml.add_representer(str, lambda dumper, data: 
            dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|') 
            if '\n' in data else dumper.represent_scalar('tag:yaml.org,2002:str', data))
    
    def save(self, path: str):
        """
        Save AgentIO instance to a YAML file.
        
        Args:
            path: Path to save YAML file
        """
        try:
            # Serialize all components
            data = {
                'dependencies': self.dependencies,
                'input_processing': self._serialize_function(self.input_processing, 'input_processing'),
                'output_processing': self._serialize_function(self.output_processing, 'output_processing')
            }
            
            # Write to YAML file with proper formatting
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w') as f:
                self._setup_yaml_multiline_representer()
                yaml.dump(data, f, default_flow_style=False, indent=2)
                
        except Exception as e:
            raise ValueError(f"Failed to save AgentIO to {path}: {e}")
    
    @classmethod
    def _deserialize_function(cls, func_data, field_name):
        """Deserialize a function from dictionary representation."""
        if func_data is None:
            return None
            
        # Check for serialization failure warning
        if func_data.get('warning'):
            print(f"Warning: {func_data['warning']}")
            return None
            
        source = func_data.get('source', '')
        func_type = func_data.get('type', '')
        
        try:
            local_namespace = {}
            
            if func_type == 'function':
                # Execute function source code and extract by name
                exec(source, globals(), local_namespace)
                function_name = func_data.get('name')
                if function_name and function_name in local_namespace:
                    return local_namespace[function_name]
                else:
                    raise ValueError(f"Function '{function_name}' not found in executed code")
                    
            elif func_type == 'lambda':
                # Evaluate lambda expression directly
                if source.startswith('lambda') and ':' in source:
                    return eval(source, globals(), local_namespace)
                else:
                    raise ValueError(f"Invalid lambda source format: {source}")
            else:
                raise ValueError(f"Unknown function type: {func_type}")
                
        except Exception as e:
            print(f"Warning: Could not reconstruct {field_name} function: {e}")
            return None
    
    @classmethod
    def load(cls, path: str):
        """
        Load an AgentIO instance from a YAML file.
        
        Args:
            path: Path to YAML file to load
            
        Returns:
            AgentIO: Loaded instance
        """
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                
            # Deserialize all components
            dependencies = data.get('dependencies', None)
            input_processing = cls._deserialize_function(data.get('input_processing'), 'input_processing')
            output_processing = cls._deserialize_function(data.get('output_processing'), 'output_processing')
            
            return cls(
                dependencies=dependencies,
                input_processing=input_processing,
                output_processing=output_processing
            )
            
        except Exception as e:
            raise ValueError(f"Failed to load AgentIO from {path}: {e}")


@dataclass
class Module:
    pass

@dataclass
class AgentModule(Module):
    agent: Agent
    io: Union[ModuleIO, Dict[str, ModuleIO]]
    progress_name: Optional[str] = None

@dataclass
class FunctionModule(Module):
    function: Callable
    io: Optional[Union[ModuleIO, Dict[str, ModuleIO]]]
    progress_name: Optional[str] = None

@dataclass
class ContentNode:
    id: str = field(default_factory=lambda: uuid4().hex)
    parent_nodes: Dict[str, List['ContentNode']] = field(default_factory=dict) # agent_name -> List[ContentNode]
    child_nodes: Dict[str, List['ContentNode']] = field(default_factory=dict) # agent_name -> List[ContentNode]
    contents: Dict[str, Any] = field(default_factory=dict)  # data_name -> data

    @property
    def is_leaf(self) -> bool:
        return len(self.child_nodes) == 0
    
    @property
    def inherited_contents(self) -> Dict[str, Any]:
        # Aggregate contents from this node and all its ancestors
        contents = {}
        nodes_to_visit = [self]
        while nodes_to_visit:
            current_node = nodes_to_visit.pop()
            for content_key, content_value in current_node.contents.items():
                if content_key not in contents:
                    contents[content_key] = content_value
                else:
                    print(f"Warning: Duplicate content key '{content_key}'. This can happen when agent name is duplicated with the input names.")
            # Add parent nodes to visit
            for parent_list in current_node.parent_nodes.values():
                nodes_to_visit.extend(parent_list)
        return deepcopy(contents)

@dataclass
class Instance:
    id: str = field(default_factory=lambda: uuid4().hex)
    content_nodes: Dict[str, ContentNode] = field(default_factory=dict)  # node_id -> ContentNode

    def __init__(self, contents):
        self.id = uuid4().hex
        root_node = ContentNode(contents=deepcopy(contents))
        self.content_nodes = {root_node.id: root_node}

    def add_content_node(self, parent_node_id: str, agent_name: str, contents: Dict[str, Any]):
        assert parent_node_id in self.content_nodes, f"Parent node ID {parent_node_id} does not exist in this instance."
        # Get parent node
        parent_node = self.content_nodes[parent_node_id]
        # Create new content node
        new_node = ContentNode(contents=deepcopy(contents))
        self.content_nodes[new_node.id] = new_node
        # Link parent->child relationship
        if agent_name not in parent_node.child_nodes:
            parent_node.child_nodes[agent_name] = []
        parent_node.child_nodes[agent_name].append(new_node)
        # Link child->parent relationship
        if agent_name not in new_node.parent_nodes:
            new_node.parent_nodes[agent_name] = []
        new_node.parent_nodes[agent_name].append(parent_node)

    @property
    def leaf_nodes(self) -> List[ContentNode]:
        # Find all leaf nodes (nodes without children)
        leaf_nodes = []
        for node in self.content_nodes.values():
            if node.is_leaf:
                leaf_nodes.append(node)
        return leaf_nodes

    @property
    def contents(self) -> Dict[str, Any]:
        # Aggregate contents from all content nodes
        contents = defaultdict(list)
        for node in self.content_nodes.values():
            for content_key, content_value in node.contents.items():
                contents[content_key].append(content_value)
        # Convert defaultdict to regular dict
        return dict(contents)
    
    def is_completed(self, agent_names: List[str]) -> bool:
        # Check if instance's contents have all required outputs from given agents
        contents = self.contents
        for agent_name in agent_names:
            if agent_name not in contents:
                return False
        return True