from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"

            # More helpful error message
            base_message = f"No relevant content found{filter_info}."
            if not course_name and not lesson_number:
                # Check if this is because no courses are loaded at all
                try:
                    total_courses = len(self.store.get_existing_course_titles())
                    if total_courses == 0:
                        base_message += " The course database appears to be empty. Please ensure documents have been loaded into the system."
                    else:
                        base_message += f" Try rephrasing your question or search in a specific course. {total_courses} courses are available."
                except:
                    base_message += " You may want to try rephrasing your question or check if courses have been loaded."

            return base_message
        
        # Format and return results
        return self._format_results(results)
    
    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # Track source for the UI with link information
            source_info = {
                'text': course_title,
                'link': None
            }
            
            if lesson_num is not None:
                source_info['text'] += f" - Lesson {lesson_num}"
                # Get lesson link from vector store
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
                if lesson_link:
                    source_info['link'] = lesson_link
            else:
                # Get course link if no specific lesson
                course_link = self.store.get_course_link(course_title)
                if course_link:
                    source_info['link'] = course_link
            
            sources.append(source_info)
            formatted.append(f"{header}\n{doc}")
        
        # Store sources for retrieval
        self.last_sources = sources
        
        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for getting complete course outlines with lesson details"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get complete course outline including course title, course link, and all lessons with their titles",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title or partial course name to get outline for (e.g. 'MCP', 'Anthropic', 'RAG')"
                    }
                },
                "required": ["course_name"]
            }
        }

    def execute(self, course_name: str) -> str:
        """
        Execute the outline tool to get course structure.

        Args:
            course_name: Course name or partial course name

        Returns:
            Formatted course outline or error message
        """
        # Step 1: Resolve course name using semantic search
        resolved_course_title = self.store._resolve_course_name(course_name)
        if not resolved_course_title:
            return f"No course found matching '{course_name}'."

        # Step 2: Get all courses metadata to find the resolved course
        all_courses = self.store.get_all_courses_metadata()
        target_course = None

        for course_meta in all_courses:
            if course_meta.get('title') == resolved_course_title:
                target_course = course_meta
                break

        if not target_course:
            return f"Course metadata not found for '{resolved_course_title}'."

        # Step 3: Format the course outline
        return self._format_course_outline(target_course)

    def _format_course_outline(self, course_meta: Dict[str, Any]) -> str:
        """Format course metadata into a structured outline"""
        course_title = course_meta.get('title', 'Unknown Course')
        course_link = course_meta.get('course_link')
        instructor = course_meta.get('instructor')
        lessons = course_meta.get('lessons', [])

        # Build the outline
        outline_parts = []

        # Course header
        header = f"**{course_title}**"
        if instructor:
            header += f" (by {instructor})"
        outline_parts.append(header)

        # Course link if available
        if course_link:
            outline_parts.append(f"Course Link: {course_link}")

        # Lessons section
        if lessons:
            outline_parts.append(f"\n**Course Outline ({len(lessons)} lessons):**")
            for lesson in lessons:
                lesson_num = lesson.get('lesson_number', '?')
                lesson_title = lesson.get('lesson_title', 'Untitled Lesson')
                lesson_line = f"{lesson_num}. {lesson_title}"
                outline_parts.append(lesson_line)
        else:
            outline_parts.append("\nNo lesson details available.")

        return "\n".join(outline_parts)


class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []