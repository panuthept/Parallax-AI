from parallax_ai.core.agent import Agent


def test_save_and_load_methods():
    # simple input and output
    agent = Agent(
        model="gpt-3.5-turbo", 
        system_prompt=(
            "Classify the sentiment of the text.\n"
            "Respond with one of the following keywords: positive, negative, neutral."
        ),
        max_tries=3,
    )
    agent.save("./temp/testing/test_agent_1.yaml")
    loaded_agent = Agent.load("./temp/testing/test_agent_1.yaml")
    print(agent.output_structure)
    print()
    print(loaded_agent.output_structure)
    assert agent.model == loaded_agent.model
    assert agent.input_structure == loaded_agent.input_structure
    assert agent.output_structure == loaded_agent.output_structure
    assert agent.system_prompt == loaded_agent.system_prompt
    assert agent.max_tries == loaded_agent.max_tries
    print("#1 test passed.")

    # structured input and output with Literal
    from typing import Literal
    agent = Agent(
        model="gpt-3.5-turbo", 
        input_structure={"label": Literal["positive", "negative", "neutral"]}, 
        output_structure={"label": Literal["positive", "negative", "neutral"]},
        system_prompt=(
            "Classify the sentiment of the text.\n"
            "Respond with one of the following keywords: positive, negative, neutral."
        ),
        max_tries=3,
    )
    agent.save("./temp/testing/test_agent_2.yaml")
    loaded_agent = Agent.load("./temp/testing/test_agent_2.yaml")
    print(agent.output_structure)
    print()
    print(loaded_agent.output_structure)
    assert agent.model == loaded_agent.model
    assert agent.input_structure == loaded_agent.input_structure
    assert agent.output_structure == loaded_agent.output_structure
    assert agent.system_prompt == loaded_agent.system_prompt
    assert agent.max_tries == loaded_agent.max_tries
    print("#2 test passed.")

    # structured input and output with normal types
    from typing import Literal
    agent = Agent(
        model="gpt-3.5-turbo", 
        input_structure={"text": str}, 
        output_structure=[{"label": str, "confidence": float}],
        system_prompt=(
            "Classify the sentiment of the text.\n"
            "Respond with one of the following keywords: positive, negative, neutral."
        ),
        max_tries=3,
    )
    agent.save("./temp/testing/test_agent_3.yaml")
    loaded_agent = Agent.load("./temp/testing/test_agent_3.yaml")
    print(agent.output_structure)
    print()
    print(loaded_agent.output_structure)
    assert agent.model == loaded_agent.model
    assert agent.input_structure == loaded_agent.input_structure
    assert agent.output_structure == loaded_agent.output_structure
    assert agent.system_prompt == loaded_agent.system_prompt
    assert agent.max_tries == loaded_agent.max_tries
    print("#3 test passed.")


if __name__ == "__main__":
    test_save_and_load_methods()