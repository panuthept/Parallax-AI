import inspect
from parallax_ai.core.multi_agent import ModuleIO, Dependency


def test_save_and_load_methods():
    agent_io = ModuleIO(
        dependency=Dependency(external_data=["data"], agent_outputs=["agent1"]),
    )
    print(agent_io)
    agent_io.save("./temp/testing/test_agent_io_1.yaml")
    loaded_agent_io = agent_io.load("./temp/testing/test_agent_io_1.yaml")
    print(agent_io)
    print()
    print(loaded_agent_io)
    assert agent_io.dependency.external_data == loaded_agent_io.dependency.external_data
    assert agent_io.dependency.agent_outputs == loaded_agent_io.dependency.agent_outputs
    print("Test #1 passed.")

    def test_output_processing(inputs, outputs, data):
        return [{**inp, **out, "data": data} for inp, out in zip(inputs, outputs)]

    agent_io = ModuleIO(
        dependency=Dependency(external_data=["data"], agent_outputs=["agent1"]),
        input_processing=lambda outputs, data: [{"output": o, "data": data["abc"]} for out in outputs for o in out],
        output_processing=test_output_processing, 
    )
    agent_io.save("./temp/testing/test_agent_io_2.yaml")
    loaded_agent_io = agent_io.load("./temp/testing/test_agent_io_2.yaml")
    assert agent_io.dependency.external_data == loaded_agent_io.dependency.external_data
    assert agent_io.dependency.agent_outputs == loaded_agent_io.dependency.agent_outputs
    
    print("Test #2 passed.")


if __name__ == "__main__":
    test_save_and_load_methods()