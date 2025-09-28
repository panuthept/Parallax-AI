import uuid
from typing import Any
from typing_validation import validate


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