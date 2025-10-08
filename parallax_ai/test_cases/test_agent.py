from parallax_ai.core.agent import Agent


def test_save_and_load_methods():
    from typing import Literal
    agent = Agent(
        model="gpt-3.5-turbo", 
        input_structure={"text": str}, 
        output_structure={"label": Literal["positive", "negative", "neutral"]},
        system_prompt=(
            "Classify the sentiment of the text.\n"
            "Respond with one of the following keywords: positive, negative, neutral."
        ),
        max_tries=3,
    )
    agent.save("./temp/testing/test_agent.yaml")
    loaded_agent = Agent.load("./temp/testing/test_agent.yaml")
    assert agent.model == loaded_agent.model
    assert agent.input_structure == loaded_agent.input_structure
    assert agent.output_structure == loaded_agent.output_structure
    assert agent.system_prompt == loaded_agent.system_prompt
    assert agent.max_tries == loaded_agent.max_tries


if __name__ == "__main__":
    test_save_and_load_methods()