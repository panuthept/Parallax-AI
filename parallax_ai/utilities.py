import uuid
import typing
from typing import Any, Literal
from typing_validation import validate
from typing import List, Dict, Tuple, Union, get_origin, get_args


def save_type_structure(type_structure: Any) -> Any:
    """
    Recursively convert type structures to a JSON-serializable format.
    """
    # Handle plain dictionaries (like {"text": str})
    if isinstance(type_structure, dict):
        return {k: save_type_structure(v) for k, v in type_structure.items()}
    
    # Handle plain lists (like [{"label": str, "confidence": float}])
    if isinstance(type_structure, list):
        return [save_type_structure(item) for item in type_structure]
    
    origin = get_origin(type_structure)
    if origin is None:
        # Handle basic types properly
        if hasattr(type_structure, '__name__'):
            return type_structure.__name__
        else:
            return str(type_structure)
    
    args = get_args(type_structure)
    if not args:
        if hasattr(type_structure, '__name__'):
            return type_structure.__name__
        else:
            return str(type_structure)
    
    if origin is list or origin is List:
        return {"list": save_type_structure(args[0])}
    elif origin is dict or origin is Dict:
        return {"dict": [save_type_structure(args[0]), save_type_structure(args[1])]}
    elif origin is tuple or origin is Tuple:
        return {"tuple": [save_type_structure(arg) for arg in args]}
    elif origin is Union:
        return {"union": [save_type_structure(arg) for arg in args]}
    else:
        return {str(origin): [save_type_structure(arg) for arg in args]}
    
def load_type_structure(data: Any) -> Any:
    """
    Recursively reconstruct type structures from a JSON-serializable format.
    """
    # Define globals for eval
    eval_globals = {
        'typing': typing,
        'Literal': Literal,
        'List': List,
        'Dict': Dict,
        'Tuple': Tuple,
        'Union': Union,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
    }
    
    if isinstance(data, str):
        # Handle basic types by name
        if data in ['str', 'int', 'float', 'bool', 'list', 'dict', 'tuple']:
            return eval(data, eval_globals)
        # Try to evaluate for other types
        try:
            return eval(data, eval_globals)
        except:
            # If eval fails, return the string as is
            return data
    
    # Handle plain lists (like [{"label": str, "confidence": float}])
    if isinstance(data, list):
        return [load_type_structure(item) for item in data]
    
    if isinstance(data, dict):
        # Check if this is a structured typing dictionary (has special keys)
        special_keys = {"list", "dict", "tuple", "union"}
        typing_keys = {k for k in data.keys() if k.startswith("typing.") or k.startswith("<class")}
        
        if any(key in special_keys for key in data.keys()) or typing_keys:
            # This is a structured typing dictionary
            for key, value in data.items():
                if key == "list":
                    return List[load_type_structure(value)]
                elif key == "dict":
                    return Dict[load_type_structure(value[0]), load_type_structure(value[1])]
                elif key == "tuple":
                    return Tuple[tuple(load_type_structure(v) for v in value)]
                elif key == "union":
                    return Union[tuple(load_type_structure(v) for v in value)]
                else:
                    origin = eval(key, eval_globals)
                    args = tuple(load_type_structure(v) for v in value)
                    return origin[args]
        else:
            # This is a plain dictionary
            return {k: load_type_structure(v) for k, v in data.items()}
    
    raise ValueError(f"Invalid data format for type structure: {data}")

def type_validation(data: Any, expected_type: type, raise_error: bool = False):
    try:
        validate(data, expected_type)
        return True
    except:
        if raise_error:
            raise TypeError(f"Type of data '{data}' is not valid: expecting {expected_type} but got {type(data)}.")
        else:
            return False
        
def generate_session_id():
    return str(uuid.uuid4())

def get_dummy_output(output_structure, default_value=None):
    """
    Dummy output based on output_structure in AgentSpec.
    Value is set to None for all fields.
    """
    def generate_dummy(structure):
        if isinstance(structure, dict):
            return {key: generate_dummy(value) for key, value in structure.items()}
        elif isinstance(structure, list):
            return [generate_dummy(item) for item in structure]
        else:
            return default_value
    return generate_dummy(output_structure)