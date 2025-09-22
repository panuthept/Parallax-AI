from dataclasses import dataclass
from typing import List, Dict, Optional
from parallax_ai import Agent, ModelContext
from dataclasses_jsonschema import JsonSchemaMixin


class GuidelineAgent(Agent):
    def __init__(self, model: str, **kwargs):
        @dataclass
        class GuidelineInputStructure(JsonSchemaMixin):
            objective: str

        @dataclass
        class GuidelineOutputStructure(JsonSchemaMixin):
            guideline: str

        super().__init__(
            model=model,
            input_structure=GuidelineInputStructure,
            output_structure=GuidelineOutputStructure,
            model_context=ModelContext(
                system_prompt=(
                    "Your task is to produce a concise, easy-to-understand guideline for data generation that meets the specified objective.\n"
                    "The guideline should encourage diverse and comprehensive data creation and comply with any provided statistical constraints or targets.\n"
                    "Write the guideline in English.\n"
                    "Plan your approach step-by-step before drafting the guideline."
                ),
            ),
            **kwargs,
        )

class WorkerAgent(Agent):
    def __init__(self, model: str, **kwargs):
        @dataclass
        class WorkerInputStructure(JsonSchemaMixin):
            guideline: str
            persona: str

        @dataclass
        class WorkerOutputStructure(JsonSchemaMixin):
            generated_data: str

        super().__init__(
            model=model,
            input_structure=WorkerInputStructure,
            output_structure=WorkerOutputStructure,
            model_context=ModelContext(
                system_prompt=(
                    "Your task is to generate data according to the provided guideline and persona.\n"
                    "Follow the guidelines strictly, and reason through each step before producing the final output."
                ),
            ),
            **kwargs,
        )

class ReviseAgent(Agent):
    def __init__(self, model: str, **kwargs):
        @dataclass
        class ReviseInputStructure(JsonSchemaMixin):
            generated_data: str
            feedback: str

        @dataclass
        class ReviseOutputStructure(JsonSchemaMixin):
            revised_data: str

        super().__init__(
            model=model,
            input_structure=ReviseInputStructure,
            output_structure=ReviseOutputStructure,
            model_context=ModelContext(
                system_prompt=(
                    "Your task is to revise the generated data based on the provided feedback.\n"
                    "Follow the guidelines strictly, and reason through each step before producing the final output."
                ),
            ),
            **kwargs,
        )

class ReviewerAgent(Agent):
    def __init__(self, model: str, **kwargs):
        @dataclass
        class ReviewerInputStructure(JsonSchemaMixin):
            criterias: List[str]
            generated_data: str

        @dataclass
        class ReviewerOutputStructure(JsonSchemaMixin):
            is_pass: bool
            feedback: str

        super().__init__(
            model=model,
            input_structure=ReviewerInputStructure,
            output_structure=ReviewerOutputStructure,
            model_context=ModelContext(
                system_prompt=(
                    "Your task is to review the generated data and determine if it meets the specified criterias.\n"
                    "If the data does not meet the specified criterias, provide feedback on how to improve it.\n"
                    "Plan your approach step-by-step before drafting the review."
                ),
            ),
            **kwargs,
        )


class DataGeneratorPipeline:
    """
    Dataset's statistic -> GuidelineAgent -> WorkerAgent -> ReviewerAgent -> AnnotatorAgent -> ResponseGeneration -> AnnotatorAgent -> Dataset -> Dataset's statistic
    """
    def __init__(
        self, 
        model: str, 
        **kwargs
    ):
        self.guideline_agent = GuidelineAgent(model=model, **kwargs)
        self.worker_agent = WorkerAgent(model=model, **kwargs)
        self.superviser_agent = ReviewerAgent(model=model, **kwargs)
        self.revise_agent = ReviseAgent(model=model, **kwargs)
        self.annotator_agent = AnnotatorAgent(model=model, **kwargs)

    def generate_data(
        self, 
        num_samples: int,
        objective: str,
        criterias: List[str],
        personas: List[str],
        dataset_statistic: Dict[str, float] = None,
    ):
        pass