import json
from typing import List
from dataclasses import dataclass


@dataclass
class Context:
    name: str
    content: str
    desc: str = None # This description is for optimizing the context when trainable is True
    title: str = None
    trainable: bool = True
    def render(self) -> str:
        return f"{self.title}:\n{self.content}" if self.title is not None else self.content


@dataclass
class ModelContext:
        input_template: str = None
        system_prompt_template: str = None
        system_prompt: List[Context] = None

        def __post_init__(self):
            if isinstance(self.system_prompt, str):
                self.system_prompt = [Context(name="system_prompt", content=self.system_prompt)]
            elif isinstance(self.system_prompt, list):
                self.system_prompt = self.system_prompt
            elif self.system_prompt is None:
                self.system_prompt = None
            else:
                raise ValueError("system_prompt must be either a string, a list of Context objects, or a SystemPrompt object")
            
        def render_system_prompt(self, output_structure = None) -> str:
            if self.system_prompt is None:
                system_prompt = None
            if self.system_prompt_template is None:
                system_prompt = "\n\n".join(content.render() for content in self.system_prompt)
            else:
                system_prompt = self.system_prompt_template.format(**{content.name: content.render() for content in self.system_prompt})

            if output_structure is not None:
                system_prompt = system_prompt + "\n\n" if system_prompt is not None else ""
                system_prompt += (
                    "The output must be JSON that matches the following schema:\n"
                    "{output_structure}"
                ).format(output_structure=json.dumps(self.get_output_schema()))
            return system_prompt
            
        def render_input(self, input_instance) -> str:
            input_instance = input_instance.to_dict()
            if self.input_template is None:
                return "\n\n".join([f"{key.capitalize()}: {value}" for key, value in input_instance.items()])
            else:
                kwargs = {key: value for key, value in input_instance.items()}
                # Remove keys that are not in the template
                for key in list(kwargs.keys()):
                    if f"{{{key}}}" not in self.input_template:
                        kwargs.pop(key)
                return self.input_template.format(**kwargs)


if __name__ == "__main__":
    model_context = ModelContext(
        system_prompt=[
            Context(name="task_definition", content="Given a topic, generate a list of people related to the topic.", title="Task Definition", trainable=False),
            Context(name="method", content="Think step by step before answering.", title="Methodology", trainable=True),
        ],
    )
    print(model_context.render_system_prompt())