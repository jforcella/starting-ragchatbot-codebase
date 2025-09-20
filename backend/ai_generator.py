import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Sequential Tool Usage Guidelines:
- **Multiple Tool Rounds**: You can use tools across up to 2 sequential rounds to provide comprehensive answers
- **Content Search Tool**: Use for questions about specific course content, lessons, or detailed educational materials
- **Outline Tool**: Use for questions about course structure, outlines, lesson lists, or course overviews
- **Strategic Tool Use**:
  * Round 1: Use tools to gather foundational information (e.g., course outlines, primary content)
  * Round 2: Use additional tools if you need complementary or comparison information
  * For complex queries requiring multiple courses: start with primary subject, then search comparison subjects
- **Tool Completion**: Stop using tools when you have sufficient information to provide a complete answer
- **Synthesize Results**: Combine information from multiple tool calls into coherent, comprehensive responses

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Simple course questions**: Use appropriate tool once and answer completely
- **Complex/comparison questions**: Use multiple rounds strategically to gather all needed information
- **No meta-commentary**:
  * Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
  * Do not mention "based on the search results", "using tools", or "in round 1/2"
  * Focus on delivering the requested information seamlessly

For Course Outlines:
- Always include the complete course title, course link (if available), and full lesson list
- Display lesson numbers and titles clearly
- Format as structured outline for easy reading

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly while being comprehensive
2. **Educational** - Maintain instructional value across all gathered information
3. **Clear** - Use accessible language and logical organization
4. **Example-supported** - Include relevant examples from multiple sources when helpful
5. **Complete** - Use multiple tool rounds when necessary to address all aspects of the query

Provide thorough, well-researched answers that synthesize information from multiple sources when appropriate.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with sequential tool usage support (up to max_rounds).

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize conversation messages
        messages = [{"role": "user", "content": query}]

        # Sequential tool calling loop
        for round_num in range(1, max_rounds + 1):
            # Prepare API call parameters for this round
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Add tools if available and not the final round
            if tools and tool_manager:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Check termination conditions
            if response.stop_reason != "tool_use" or not tool_manager:
                # No tool use - return response immediately
                return response.content[0].text

            # Execute tools and update conversation for next round
            messages = self._execute_tools_and_update_messages(
                messages, response, tool_manager, round_num
            )

            # If this was the final round, make one last call without tools
            if round_num == max_rounds:
                final_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": system_content
                }
                # No tools in final call to get synthesis response
                final_response = self.client.messages.create(**final_params)
                return final_response.content[0].text

        # Fallback (should not reach here with current logic)
        return "Maximum tool rounds exceeded"

    def _execute_tools_and_update_messages(self, messages: List, response, tool_manager, round_num: int) -> List:
        """
        Execute tools and maintain conversation context for next round.

        Args:
            messages: Current conversation messages
            response: API response containing tool use requests
            tool_manager: Manager to execute tools
            round_num: Current round number for error context

        Returns:
            Updated messages list with tool execution results
        """
        # Add Claude's tool use response to conversation
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })

                except Exception as e:
                    # Tool execution failed - add error result but continue
                    error_msg = f"Tool execution failed in round {round_num}: {str(e)}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": error_msg
                    })

        # Add tool results as user message for next round
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        [LEGACY METHOD - Used for backward compatibility with single-round tool execution]

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text