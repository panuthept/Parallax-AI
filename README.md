# Parallax-AI

## Installation

You can install Parallax using pip:

```bash
pip install parallax-ai
```

### Usage Example
```python
from parallax_ai import Service, OutputComposer
from parallax_ai.modules import AgentSpec, AgentModule, ClassificationModule, SwitchModule, LambdaModule, ModuleInterface


worker_nodes = "path/to/your/worker_nodes.json"
# Or define worker nodes directly
worker_nodes = {
    "google/gemma-3-27b-it": [
        {"api_key": "your_api_key_here", "base_url": "https://api.example.com/v1"},
        {"api_key": "your_api_key_here", "base_url": "https://api.example.com/v1"},
        {"api_key": "your_api_key_here", "base_url": "https://api.example.com/v1"}
    ]
}

service = Service(
    name="TranslationService",
    worker_nodes=worker_nodes,
    modules=[
        AgentModule(
            name="translator",
            spec=AgentSpec(
                model_name="google/gemma-3-27b-it",
                system_prompt="You are a helpful assistant that translates English to French.",
                input_structure={"text": str, "persona_info": str},
                output_structure={"translation": str},
            ),
            interface=Interface(
                dependencies=["text", "persona_infos"],
                input_processing=lambda deps: [
                    {
                        "text": deps["text"],
                        "persona_info": persona_info,
                    } for persona_info in deps["persona_infos"]
                ],
                output_processing=lambda inputs, outputs: {
                    "translations": [out["translation"] for out in outputs]
                },
            ),
        ),
        ClassificationModule(
            name="reviewer",
            spec=AgentSpec(
                model_name="google/gemma-3-27b-it",
                system_prompt="You are a helpful assistant that reviews translations for accuracy.",
                input_structure={"original": str, "translation": str},
                output_structure={"need_revision": bool},
            ),
            n=10,
            interface=ModuleInterface(
                dependencies=["text", "translations"],
                input_processing=lambda deps: [
                    {
                        "original": deps["text"],
                        "translation": translation,
                    } for translation in deps["translations"]
                ],
                output_processing=lambda inputs, outputs: {
                    "reviews": [out["need_revision"] for out in outputs]
                },
            ),
        ),
        SwitchModule(
            name="reviser",
            condition_key="need_revision",
            cases={
                True: AgentModule(
                    spec=AgentSpec(
                        model_name="google/gemma-3-27b-it",
                        system_prompt="You are a helpful assistant that revises translations if needed.",
                        input_structure={"original": str, "translation": str},
                        output_structure={"revised_translation": str},
                    ),
                ),
                False: LambdaModule(
                    function=lambda inp: {"revised_translation": None}
                ),
            },
            interface=ModuleInterface(
                dependencies=["text", "translations", "reviews"],
                input_processing=lambda deps: [
                    {
                        "original": deps["text"],
                        "translation": translation,
                        "need_revision": review,
                    } for translation, review in zip(deps["translations"], deps["reviews"])
                ],
                output_processing=lambda inputs, outputs: {
                    "revised_translations": [out["revised_translation"] for out in outputs]
                },
            ),
        ),
    ],
    output_composers=[
        OutputComposer(
            name="no_revision",
            dependencies=["text", "translations", "reviews"],
            condition=lambda deps: all(not review for review in deps["reviews"]),
            compose=lambda deps: {
                "original": deps["text"],
                "translations": deps["translations"],
            },
        ),
        OutputComposer(
            name="revision",
            dependencies=["text", "translations", "reviews", "revised_translations"],
            compose=lambda deps: {
                "original": deps["text"],
                "translations": [
                    revised if review else translation
                    for translation, review, revised in zip(
                        deps["translations"],
                        deps["reviews"],
                        deps["revised_translations"],
                    ),
                ],
            },
        ),
    ],
)
outputs = service.run(
    inputs=[
        {
            "text": "Hello, how are you?",
            "persona_infos": ["You are a friendly and polite person.", "You like to use formal language."],
        },
    ]
)
print(outputs)
# Expected Output:
[
    {
        "original": "Hello, how are you?", 
        "translations": ["Bonjour, comment ça va?", "Salut, comment ça va?"]
        "metadata": {"service_name": "TranslationService", "output_name": "no_revision"}
    }
]
```