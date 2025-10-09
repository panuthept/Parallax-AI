
import yaml
import dill
import types
import inspect
from ..agent import Agent
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Union, Optional, Any


@dataclass
class Dependency:
    external_data: Optional[List[str]] = None
    agent_outputs: Optional[List[str]] = None


@dataclass
class ModuleIO:
    dependency: Optional[Dependency] = None
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
    
    def _serialize_dependency(self):
        """Serialize dependency to a dictionary."""
        if self.dependency is None:
            return None
            
        return {
            'external_data': self.dependency.external_data,
            'agent_outputs': self.dependency.agent_outputs
        }
    
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
                'dependency': self._serialize_dependency(),
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
    def _deserialize_dependency(cls, data):
        """Deserialize dependency from dictionary."""
        dependency_data = data.get('dependency')
        if dependency_data is None:
            return None
            
        return Dependency(
            external_data=dependency_data.get('external_data'),
            agent_outputs=dependency_data.get('agent_outputs')
        )
    
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
            dependency = cls._deserialize_dependency(data)
            input_processing = cls._deserialize_function(data.get('input_processing'), 'input_processing')
            output_processing = cls._deserialize_function(data.get('output_processing'), 'output_processing')
            
            return cls(
                dependency=dependency,
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
    io: Optional[Union[ModuleIO, Dict[str, ModuleIO]]]

@dataclass
class FunctionModule(Module):
    function: Callable
    io: Optional[Union[ModuleIO, Dict[str, ModuleIO]]]


@dataclass
class Package:
    id: str = field(default_factory=lambda: uuid4().hex)
    agent_inputs: Dict[str, Any] = field(default_factory=dict)      # agent_name -> inputs
    agent_outputs: Dict[str, Any] = field(default_factory=dict)     # agent_name -> outputs
    external_data: Dict[str, Any] = field(default_factory=dict)     # data_name -> data_value