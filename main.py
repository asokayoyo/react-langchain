"""This project for diving deep in to tool binding with LLMs."""

from callbacks import AgentCallbackHandler
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

load_dotenv()


### Tools
@tool
def text_length_tool(text: str) -> int:
    """Return the length of the input text."""
    print(f"text_length_tool: Calculating length of text: {text}")
    return len(text.strip("\n").strip('"'))


def main():
    """Main module for react-langchain application."""
    print("Hello from react-langchain with .bind_tools()!")

    tools = [text_length_tool]

    # Create LLM and bind tools to it
    llm = ChatOpenAI(
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )
    llm_with_tools = llm.bind_tools(tools)

    # Message history for conversation
    messages = [
        HumanMessage(content="What is the length in characters of the text CAT ?")
    ]

    # Agent loop
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        print(f"AI Response: {response}")
        messages.append(response)

        # Check if there are tool calls
        if not response.tool_calls:
            # No tool calls means we have a final answer
            print("\n=== Final Answer ===")
            print(response.content)
            break

        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            print(f"\nTool Call: {tool_name}")
            print(f"Tool Args: {tool_args}")

            # Find and execute the tool
            selected_tool = {tool.name: tool for tool in tools}[tool_name]
            tool_output = selected_tool.invoke(tool_args)

            print(f"Tool Output: {tool_output}")

            # Add tool message to history
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))

    if iteration >= max_iterations:
        print(f"\nReached max iterations ({max_iterations})")


if __name__ == "__main__":
    main()
