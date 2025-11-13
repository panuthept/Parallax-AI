from ..agent_module import ModelSpec
from dataclasses import dataclass, field
from .base_module import BaseGuardModule


@dataclass
class ShieldGemmaModule(BaseGuardModule):
    spec: ModelSpec = field(default_factory=lambda: ModelSpec(model_name="google/shieldgemma-9b")) 
    max_retries: int = 10
    representative_token_index: int = 0
    representative_tokens: dict = field(default_factory=lambda: 
        {
            "No": "Safe",
            "Yes": "Harmful",
        }
    )
    unsafe_categories = (
        '"No Dangerous Content": The prompt shall not contain or seek generation of content that harming oneself and/or others (e.g., accessing or building firearms and explosive devices, promotion of terrorism, instructions for suicide).\n'
        '* "No Harassment": The prompt shall not contain or seek generation of content that is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence).\n'
        '* "No Hate Speech": The prompt shall not contain or seek generation of content that targeting identity and/or protected attributes (e.g., racial slurs, promotion of discrimination, calls to violence against protected groups) or dehumanizing/belittling/vilifying on the bases of race, ethnicity, religion, disability, age, nationality, veteran status, sexual orientation, gender, gender identity, caste, or any other protected status. We do allow content that includes a reference to a protected individual or group in a positive manner.\n'
        '* "No Sexually Explicit Information": The prompt shall not contain or seek generation of content that contains references to sexual acts or other lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal). Medical or scientific terms related to human anatomy or sex education are permitted.\n'
    )

    def get_safeguard_input(self, module_input: dict) -> dict:
        task = "prompt_classification"
        messages = [
            {"role": "system", "content": self.unsafe_categories},
            {"role": "user", "content": module_input["prompt"]}
        ] 
        if module_input.get("response") is not None:
            task = "response_classification"
            messages.append({"role": "assistant", "content": module_input["response"]})
        return {"messages": messages, "task": task}