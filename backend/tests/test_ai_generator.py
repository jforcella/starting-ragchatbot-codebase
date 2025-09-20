import pytest
from unittest.mock import Mock, patch, MagicMock

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator class"""

    @pytest.fixture
    def ai_generator(self, mock_config):
        """Create AIGenerator with test configuration"""
        return AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager"""
        mock_manager = Mock()
        mock_manager.execute_tool.return_value = "Tool execution result"
        return mock_manager

    @pytest.fixture
    def sample_tools(self):
        """Sample tool definitions"""
        return [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        ]

    def test_initialization(self, mock_config):
        """Test AIGenerator initialization"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)

            assert ai_gen.model == mock_config.ANTHROPIC_MODEL
            assert ai_gen.base_params["model"] == mock_config.ANTHROPIC_MODEL
            assert ai_gen.base_params["temperature"] == 0
            assert ai_gen.base_params["max_tokens"] == 800
            mock_anthropic.assert_called_once_with(api_key=mock_config.ANTHROPIC_API_KEY)

    def test_generate_response_simple(self, ai_generator, mock_anthropic_client):
        """Test simple response generation without tools"""
        ai_generator.client = mock_anthropic_client

        response = ai_generator.generate_response("What is Python?")

        assert response == "This is a test response"
        mock_anthropic_client.messages.create.assert_called_once()

        # Check that the call was made with correct parameters
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"][0]["content"] == "What is Python?"
        assert "tools" not in call_args

    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_client):
        """Test response generation with conversation history"""
        ai_generator.client = mock_anthropic_client

        history = "Previous conversation context"
        response = ai_generator.generate_response("Follow-up question", conversation_history=history)

        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert history in call_args["system"]

    def test_generate_response_with_tools_no_tool_use(self, ai_generator, mock_anthropic_client, sample_tools, mock_tool_manager):
        """Test response with tools available but no tool use"""
        ai_generator.client = mock_anthropic_client

        response = ai_generator.generate_response(
            "General question",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        assert response == "This is a test response"

        # Verify tools were passed to the API call
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == sample_tools
        assert call_args["tool_choice"] == {"type": "auto"}

    def test_generate_response_with_tool_use(self, ai_generator, mock_tool_manager, sample_tools):
        """Test response generation with tool use"""
        # Create mock tool response
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test query"}
        mock_tool_block.id = "tool_123"

        # Create mock initial response (tool use)
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        mock_initial_response.content = [mock_tool_block]

        # Create mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response after tool use")]

        # Mock client to return initial response, then final response
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]

        ai_generator.client = mock_client

        response = ai_generator.generate_response(
            "Search for something",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        assert response == "Final response after tool use"

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

    def test_handle_tool_execution_single_tool(self, ai_generator, mock_tool_manager):
        """Test handling of single tool execution"""
        # Create mock tool use block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "Python basics"}
        mock_tool_block.id = "tool_456"

        # Create mock initial response
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]

        # Create base parameters
        base_params = {
            "messages": [{"role": "user", "content": "What is Python?"}],
            "system": "You are a helpful assistant",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800
        }

        # Mock final API response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Python is a programming language")]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response

        ai_generator.client = mock_client

        result = ai_generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )

        assert result == "Python is a programming language"

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )

        # Verify final API call was made with tool results
        mock_client.messages.create.assert_called_once()
        final_call_args = mock_client.messages.create.call_args[1]

        # Check that tool results were added to messages
        messages = final_call_args["messages"]
        assert len(messages) == 3  # original user message + assistant response + tool results
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_456"
        assert messages[2]["content"][0]["content"] == "Tool execution result"

    def test_handle_tool_execution_multiple_tools(self, ai_generator, mock_tool_manager):
        """Test handling multiple tool executions"""
        # Create multiple mock tool blocks
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.input = {"query": "Python"}
        mock_tool_block1.id = "tool_1"

        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "get_course_outline"
        mock_tool_block2.input = {"course_name": "Python Course"}
        mock_tool_block2.id = "tool_2"

        # Create mock initial response with multiple tools
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block1, mock_tool_block2]

        base_params = {
            "messages": [{"role": "user", "content": "Tell me about Python course"}],
            "system": "You are a helpful assistant",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800
        }

        # Mock tool execution results
        mock_tool_manager.execute_tool.side_effect = ["Search result", "Outline result"]

        # Mock final API response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Combined tool results")]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response

        ai_generator.client = mock_client

        result = ai_generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )

        assert result == "Combined tool results"

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="Python")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Python Course")

        # Verify final API call includes results from both tools
        final_call_args = mock_client.messages.create.call_args[1]
        tool_results = final_call_args["messages"][2]["content"]
        assert len(tool_results) == 2
        assert tool_results[0]["content"] == "Search result"
        assert tool_results[1]["content"] == "Outline result"

    def test_handle_tool_execution_non_tool_blocks(self, ai_generator, mock_tool_manager):
        """Test that non-tool blocks are ignored during tool execution"""
        # Create mixed content blocks
        mock_text_block = Mock()
        mock_text_block.type = "text"

        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_123"

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_text_block, mock_tool_block]

        base_params = {
            "messages": [{"role": "user", "content": "Test"}],
            "system": "System prompt",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800
        }

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final result")]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response

        ai_generator.client = mock_client

        result = ai_generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )

        # Only one tool should have been executed (non-tool block ignored)
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="test")

        # Only one tool result should be in the final call
        final_call_args = mock_client.messages.create.call_args[1]
        tool_results = final_call_args["messages"][2]["content"]
        assert len(tool_results) == 1

    def test_handle_tool_execution_no_tools(self, ai_generator, mock_tool_manager):
        """Test tool execution handling when no tool blocks are present"""
        mock_text_block = Mock()
        mock_text_block.type = "text"

        mock_initial_response = Mock()
        mock_initial_response.content = [mock_text_block]

        base_params = {
            "messages": [{"role": "user", "content": "Test"}],
            "system": "System prompt",
            "model": "claude-sonnet-4-20250514",
            "temperature": 0,
            "max_tokens": 800
        }

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="No tools used")]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response

        ai_generator.client = mock_client

        result = ai_generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )

        # No tools should have been executed
        mock_tool_manager.execute_tool.assert_not_called()

        # Final message should not include tool results
        final_call_args = mock_client.messages.create.call_args[1]
        messages = final_call_args["messages"]
        # Should have original + assistant response, no tool results
        assert len(messages) == 2

    def test_system_prompt_content(self, ai_generator):
        """Test that system prompt contains expected content"""
        system_prompt = ai_generator.SYSTEM_PROMPT

        # Check for key instruction elements
        assert "course materials and educational content" in system_prompt.lower()
        assert "content search tool" in system_prompt.lower()
        assert "outline tool" in system_prompt.lower()
        assert "multiple tool rounds" in system_prompt.lower()
        assert "up to 2 sequential rounds" in system_prompt.lower()
        assert "strategic tool use" in system_prompt.lower()
        assert "brief, concise and focused" in system_prompt.lower()

    @patch('anthropic.Anthropic')
    def test_anthropic_api_error_handling(self, mock_anthropic_class, ai_generator, sample_tools, mock_tool_manager):
        """Test handling of Anthropic API errors"""
        # Mock the Anthropic client to raise an exception
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic_class.return_value = mock_client
        ai_generator.client = mock_client

        # This should raise the exception (not caught by AIGenerator)
        with pytest.raises(Exception, match="API Error"):
            ai_generator.generate_response(
                "Test query",
                tools=sample_tools,
                tool_manager=mock_tool_manager
            )

    def test_base_params_configuration(self, ai_generator, mock_config):
        """Test that base parameters are configured correctly"""
        base_params = ai_generator.base_params

        assert base_params["model"] == mock_config.ANTHROPIC_MODEL
        assert base_params["temperature"] == 0
        assert base_params["max_tokens"] == 800

    def test_generate_response_preserves_message_structure(self, ai_generator, mock_anthropic_client):
        """Test that message structure is preserved correctly"""
        ai_generator.client = mock_anthropic_client

        query = "What is machine learning?"
        history = "Previous conversation about AI"

        ai_generator.generate_response(query, conversation_history=history)

        call_args = mock_anthropic_client.messages.create.call_args[1]

        # Check message structure
        messages = call_args["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == query

        # Check system prompt includes history
        system_content = call_args["system"]
        assert ai_generator.SYSTEM_PROMPT in system_content
        assert history in system_content

    def test_sequential_tool_calling_single_round_completion(self, ai_generator, mock_tool_manager, sample_tools):
        """Test query that completes in first round (no tool_use)"""
        # Mock client to return direct response without tool_use
        mock_client = Mock()
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct answer without tools")]

        mock_client.messages.create.return_value = mock_response
        ai_generator.client = mock_client

        response = ai_generator.generate_response(
            "What is Python?",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Should return direct response after single API call
        assert response == "Direct answer without tools"
        mock_client.messages.create.assert_called_once()
        mock_tool_manager.execute_tool.assert_not_called()

    def test_sequential_tool_calling_two_rounds(self, ai_generator, mock_tool_manager, sample_tools):
        """Test full two-round sequential tool calling"""
        mock_client = Mock()

        # Round 1: Tool use response
        mock_tool_block_1 = Mock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "get_course_outline"
        mock_tool_block_1.input = {"course_name": "MCP Course"}
        mock_tool_block_1.id = "tool_1"

        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block_1]

        # Round 2: Another tool use response
        mock_tool_block_2 = Mock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "search_course_content"
        mock_tool_block_2.input = {"query": "lesson 4 content"}
        mock_tool_block_2.id = "tool_2"

        mock_response_2 = Mock()
        mock_response_2.stop_reason = "tool_use"
        mock_response_2.content = [mock_tool_block_2]

        # Final response (after max rounds)
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Complete synthesis of both tool results")]

        # Mock tool execution results
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline result",
            "Lesson 4 content result"
        ]

        # Set up API calls sequence
        mock_client.messages.create.side_effect = [
            mock_response_1,  # Round 1
            mock_response_2,  # Round 2
            mock_final_response  # Final synthesis
        ]

        ai_generator.client = mock_client

        response = ai_generator.generate_response(
            "Compare lesson 4 of MCP course with similar content",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )

        # Should have made 3 API calls (round 1, round 2, final synthesis)
        assert mock_client.messages.create.call_count == 3

        # Should have executed 2 tools
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP Course")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="lesson 4 content")

        # Should return final synthesized response
        assert response == "Complete synthesis of both tool results"

    def test_sequential_tool_calling_early_termination(self, ai_generator, mock_tool_manager, sample_tools):
        """Test termination when Claude stops using tools before max rounds"""
        mock_client = Mock()

        # Round 1: Tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "Python basics"}
        mock_tool_block.id = "tool_1"

        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block]

        # Round 2: Direct response (no more tool use)
        mock_response_2 = Mock()
        mock_response_2.stop_reason = "end_turn"
        mock_response_2.content = [Mock(text="Complete answer based on search results")]

        mock_tool_manager.execute_tool.return_value = "Search results"

        # Set up API calls
        mock_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        ai_generator.client = mock_client

        response = ai_generator.generate_response(
            "What is Python?",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )

        # Should have made only 2 API calls (terminated early)
        assert mock_client.messages.create.call_count == 2

        # Should have executed only 1 tool
        mock_tool_manager.execute_tool.assert_called_once()

        # Should return the early termination response
        assert response == "Complete answer based on search results"

    def test_sequential_tool_calling_tool_execution_error(self, ai_generator, mock_tool_manager, sample_tools):
        """Test handling of tool execution errors in sequential calling"""
        mock_client = Mock()

        # Round 1: Tool use that fails
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_1"

        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block]

        # Round 2: Direct response after tool error
        mock_response_2 = Mock()
        mock_response_2.stop_reason = "end_turn"
        mock_response_2.content = [Mock(text="Response despite tool error")]

        # Mock tool execution failure
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        mock_client.messages.create.side_effect = [mock_response_1, mock_response_2]
        ai_generator.client = mock_client

        response = ai_generator.generate_response(
            "Test query",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Should continue execution despite tool error
        assert response == "Response despite tool error"
        assert mock_client.messages.create.call_count == 2

        # Verify error message was included in conversation
        second_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_args["messages"]

        # Should have user message, assistant tool use, and user tool result (with error)
        assert len(messages) >= 3
        tool_result_message = messages[2]
        assert tool_result_message["role"] == "user"
        assert "Tool execution failed in round 1" in str(tool_result_message["content"])

    def test_sequential_tool_calling_max_rounds_parameter(self, ai_generator, mock_tool_manager, sample_tools):
        """Test that max_rounds parameter is respected"""
        mock_client = Mock()

        # Create continuous tool use responses
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

        # Final response after max rounds
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"  # Non-tool_use stop reason
        mock_final_content = Mock()
        mock_final_content.text = "Final response after max rounds"
        mock_final_response.content = [mock_final_content]

        # Set up responses for 2 rounds + final synthesis call
        mock_client.messages.create.side_effect = [
            create_tool_response("tool_1"),  # Round 1
            create_tool_response("tool_2"),  # Round 2
            mock_final_response              # Final synthesis call (after max_rounds reached)
        ]

        mock_tool_manager.execute_tool.return_value = "Tool result"
        ai_generator.client = mock_client

        response = ai_generator.generate_response(
            "Test query",
            tools=sample_tools,
            tool_manager=mock_tool_manager,
            max_rounds=2  # Limit to 2 rounds
        )

        # Should have made exactly 3 calls: round 1, round 2, final synthesis
        assert mock_client.messages.create.call_count == 3
        assert response == "Final response after max rounds"

    def test_sequential_tool_calling_conversation_history_preservation(self, ai_generator, mock_tool_manager, sample_tools):
        """Test that conversation history is preserved across rounds"""
        mock_client = Mock()

        # Mock single round with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.input = {"query": "test"}
        mock_tool_block.id = "tool_1"

        mock_response_1 = Mock()
        mock_response_1.stop_reason = "tool_use"
        mock_response_1.content = [mock_tool_block]

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response")]

        mock_client.messages.create.side_effect = [mock_response_1, mock_final_response]
        mock_tool_manager.execute_tool.return_value = "Tool result"
        ai_generator.client = mock_client

        response = ai_generator.generate_response(
            "Test query",
            conversation_history="Previous conversation context",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Check that all API calls included conversation history
        for call in mock_client.messages.create.call_args_list:
            call_args = call[1]
            system_content = call_args["system"]
            assert "Previous conversation context" in system_content

        assert response == "Final response"