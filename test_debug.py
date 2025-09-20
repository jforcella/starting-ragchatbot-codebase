#!/usr/bin/env python3
"""Debug script for sequential tool calling"""

import sys
import os
sys.path.append('./backend')

from unittest.mock import Mock
from ai_generator import AIGenerator

# Create AI generator
ai_gen = AIGenerator("test-key", "claude-sonnet-4-20250514")

# Mock client
mock_client = Mock()

# Create tool response
def create_tool_response(tool_id):
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.input = {"query": f"query_{tool_id}"}
    mock_tool_block.id = tool_id

    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    mock_response.content = [mock_tool_block]
    return mock_response

# Final response
mock_final_response = Mock()
mock_final_response.content = [Mock(text="Final response test")]

print("Testing Mock structure...")
print(f"Final response content: {mock_final_response.content}")
print(f"First content item: {mock_final_response.content[0]}")
print(f"Text attribute: {mock_final_response.content[0].text}")
print(f"Text type: {type(mock_final_response.content[0].text)}")

# Set up client responses
mock_client.messages.create.side_effect = [
    create_tool_response("tool_1"),
    create_tool_response("tool_2"),
    mock_final_response
]

# Mock tool manager
mock_tool_manager = Mock()
mock_tool_manager.execute_tool.return_value = "Tool result"

# Mock tools
sample_tools = [{"name": "search_course_content"}]

# Set client
ai_gen.client = mock_client

print("\nTesting generate_response...")
try:
    response = ai_gen.generate_response(
        "Test query",
        tools=sample_tools,
        tool_manager=mock_tool_manager,
        max_rounds=2
    )

    print(f"Response: {response}")
    print(f"Response type: {type(response)}")
    print(f"API calls made: {mock_client.messages.create.call_count}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()