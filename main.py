"""This project for diving deep in to the react agent. How it works and how to extend it."""

from callbacks import AgentCallbackHandler
from langchain_classic.agents.format_scratchpad.log import format_log_to_str
from dotenv import load_dotenv
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.schema import AgentAction, AgentFinish
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.tools import render_text_description, Tool, tool
from langchain_openai import ChatOpenAI
from typing import Union

load_dotenv()


### Tools
@tool
def text_length_tool(text: str) -> int:
    """Return the length of the input text."""
    print(f"text_length_tool: Calculating length of text: {text}")
    return len(text.strip("\n").strip('"'))


def find_tool_by_name(tools, name: str) -> Tool:
    """Find a tool by its name from a list of tools."""
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool with name {name} not found.")


def main():
    """Main module for react-langchain application."""
    print("Hello from react-langchain!")

    tools = [text_length_tool]
    template = """
                Answer the following questions as best you can. You have access to the following tools:

                {tools}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Begin!

                Question: {input}
                Thought: {agent_scratchpad}
                """
    prompt = PromptTemplate(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )
    llm = ChatOpenAI(
        temperature=0,
        stop=["\nObservation", "Observation:", "Observation"],
        callbacks=[AgentCallbackHandler()],
    )
    intermediate_steps = []
    input_extraction = {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    agent = input_extraction | prompt | llm | ReActSingleInputOutputParser()

    agent_step: Union[AgentAction, AgentFinish] = None
    while not isinstance(agent_step, AgentFinish):
        agent_step = agent.invoke(
            {
                "input": "What is the length in characters of the text CAT ?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print("Agent Result:")
        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            print(f"Tool used: {tool_name}")
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            observation = tool_to_use.func(str(tool_input))
            print(f"Tool Observation: {observation:}")
            intermediate_steps.append((agent_step, str(observation)))

    print("Final Answer:")
    print(agent_step.return_values["output"])


if __name__ == "__main__":
    main()
