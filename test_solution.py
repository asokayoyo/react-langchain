"""Simple test to verify the tool calling implementation."""

import sys
import os

# Import from execise.py
from execise import (
    implement_set_api_key,
    implement_create_model_with_tools,
    implement_check_for_tool_calls,
    implement_execute_tool_call,
    implement_run_agent_with_tool_calling,
    text_length_tool,
    AIMessage,
)


def test_implementation():
    print("Testing Tool Calling Implementation")
    print("=" * 50)

    # Test 1: Set API Key
    print("\n1. Testing implement_set_api_key...")
    implement_set_api_key("test_key_123")
    assert os.environ.get("OPENAI_API_KEY") == "test_key_123"
    print("   ✅ API key set successfully")

    # Test 2: Create model with tools
    print("\n2. Testing implement_create_model_with_tools...")
    tools = [text_length_tool]
    model = implement_create_model_with_tools(tools)
    assert model is not None
    assert len(model.tools) == 1
    print("   ✅ Model created with tools bound")

    # Test 3: Check for tool calls
    print("\n3. Testing implement_check_for_tool_calls...")
    msg_with_tools = AIMessage(content="", tool_calls=[{"name": "test"}])
    msg_without_tools = AIMessage(content="Just text")
    assert implement_check_for_tool_calls(msg_with_tools) == True
    assert implement_check_for_tool_calls(msg_without_tools) == False
    print("   ✅ Tool call detection working")

    # Test 4: Execute tool call
    print("\n4. Testing implement_execute_tool_call...")
    tool_call = {"name": "get_text_length", "args": {"text": "HELLO"}, "id": "call_123"}
    result = implement_execute_tool_call(tool_call, tools)
    assert result == "5"
    print(f"   ✅ Tool executed successfully, result: {result}")

    # Test 5: Run agent with tool calling
    print("\n5. Testing implement_run_agent_with_tool_calling...")
    user_input = "What is the length of the word: DOG"
    final_result = implement_run_agent_with_tool_calling(model, user_input, tools)
    assert final_result == "3"
    print(f"   ✅ Agent executed successfully, result: {final_result}")

    print("\n" + "=" * 50)
    print("🎉 All tests passed! Implementation is correct!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        test_implementation()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
