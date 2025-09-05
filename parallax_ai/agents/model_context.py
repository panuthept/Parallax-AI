import json
from typing import List
from dataclasses import dataclass


@dataclass
class Field:
    name: str
    content: str
    desc: str = None # This description is for optimizing the context when trainable is True
    title: str = None
    trainable: bool = True

    def render(self, training: bool = False) -> str:
        desc = self.desc if training and self.desc is not None else ""
        context = f"{self.title}: {desc}\n{self.content}" if self.title is not None else f"{desc}\n{self.content}".strip()
        if training:
            context = (
                "<start><name={name}, trainable={trainable}>\n"
                "{context}\n"
                "<end>"
            ).format(name=self.name, trainable=self.trainable, context=context)
        return context


@dataclass
class ModelContext:
        input_template: str = None
        system_prompt_template: str = None
        system_prompt: List[Field] = None

        def to_json(self, path):
            data = {
                "input_template": self.input_template,
                "system_prompt_template": self.system_prompt_template,
                "system_prompt": [context.__dict__ for context in self.system_prompt] if self.system_prompt is not None else None,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        
        @classmethod
        def from_json(cls, path):
            with open(path, "r") as f:
                data = json.load(f)
            return ModelContext(
                input_template=data.get("input_template", None),
                system_prompt_template=data.get("system_prompt_template", None),
                system_prompt=[Field(**context) for context in data.get("system_prompt", [])] if data.get("system_prompt", None) is not None else None,
            )

        def __post_init__(self):
            if isinstance(self.system_prompt, str):
                self.system_prompt = [Field(name="system_prompt", content=self.system_prompt)]
            elif isinstance(self.system_prompt, list):
                self.system_prompt = self.system_prompt
            elif self.system_prompt is None:
                self.system_prompt = None
            else:
                raise ValueError("system_prompt must be either a string, a list of Context objects, or a SystemPrompt object")
            
        def render_system_prompt(self, output_structure = None, training: bool = False) -> str:
            if self.system_prompt is None:
                system_prompt = None
            if self.system_prompt_template is None:
                system_prompt = "\n\n".join(content.render(training=training) for content in self.system_prompt)
            else:
                system_prompt = self.system_prompt_template.format(**{content.name: content.render(training=training) for content in self.system_prompt})

            if output_structure is not None:
                output_schema = output_structure.json_schema() if output_structure is not None else None
                output_schema.pop("$schema", None)
                output_schema.pop("description", None)

                system_prompt = system_prompt + "\n\n" if system_prompt is not None else ""
                system_prompt += (
                    "The output must be JSON that matches the following schema:\n"
                    "{output_structure}"
                ).format(output_structure=json.dumps(output_schema, indent=2))
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
            Field(name="task_definition", content="Given a topic, generate a list of people related to the topic.", title="Task Definition", trainable=False),
            Field(name="method", content="Think step by step before answering.", title="Methodology", trainable=True),
        ],
    )
    print(model_context.render_system_prompt())