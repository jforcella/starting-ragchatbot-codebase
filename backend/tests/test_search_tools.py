import pytest
from unittest.mock import Mock, patch
import json

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults
from models import Course, Lesson


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create CourseSearchTool with mocked vector store"""
        return CourseSearchTool(mock_vector_store)

    def test_get_tool_definition(self, search_tool):
        """Test that tool definition is correctly structured"""
        definition = search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]

    def test_execute_successful_search(self, search_tool, mock_vector_store, search_results_success):
        """Test successful search execution"""
        mock_vector_store.search.return_value = search_results_success
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = search_tool.execute("test query")

        assert "Software Testing Course" in result
        assert "This is lesson 1 content" in result
        assert len(search_tool.last_sources) == 2

    def test_execute_with_course_filter(self, search_tool, mock_vector_store, search_results_success):
        """Test search with course name filter"""
        mock_vector_store.search.return_value = search_results_success

        result = search_tool.execute("test query", course_name="Testing Course")

        # Verify search was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Testing Course",
            lesson_number=None
        )
        assert "Software Testing Course" in result

    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store, search_results_success):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = search_results_success

        result = search_tool.execute("test query", lesson_number=1)

        # Verify search was called with lesson filter
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=1
        )
        assert "Software Testing Course" in result

    def test_execute_with_both_filters(self, search_tool, mock_vector_store, search_results_success):
        """Test search with both course and lesson filters"""
        mock_vector_store.search.return_value = search_results_success

        result = search_tool.execute("test query", course_name="Testing Course", lesson_number=1)

        # Verify search was called with both filters
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Testing Course",
            lesson_number=1
        )

    def test_execute_empty_results(self, search_tool, mock_vector_store, search_results_empty):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = search_results_empty

        result = search_tool.execute("nonexistent query")

        assert "No relevant content found" in result
        assert len(search_tool.last_sources) == 0

    def test_execute_empty_results_with_filters(self, search_tool, mock_vector_store, search_results_empty):
        """Test empty results with filters show appropriate message"""
        mock_vector_store.search.return_value = search_results_empty

        result = search_tool.execute("query", course_name="NonExistent", lesson_number=99)

        assert "No relevant content found in course 'NonExistent' in lesson 99" in result

    def test_execute_with_error(self, search_tool, mock_vector_store, search_results_error):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = search_results_error

        result = search_tool.execute("test query")

        assert "Test error: ChromaDB connection failed" in result
        assert len(search_tool.last_sources) == 0

    def test_format_results_with_links(self, search_tool, mock_vector_store):
        """Test result formatting includes links when available"""
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[{
                "course_title": "Test Course",
                "lesson_number": 1,
                "chunk_index": 0
            }],
            distances=[0.2]
        )

        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = search_tool._format_results(search_results)

        assert "[Test Course - Lesson 1]" in result
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]['link'] == "https://example.com/lesson1"

    def test_format_results_without_lesson_number(self, search_tool, mock_vector_store):
        """Test result formatting when lesson number is not available"""
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[{
                "course_title": "Test Course",
                "chunk_index": 0
            }],
            distances=[0.2]
        )

        mock_vector_store.get_course_link.return_value = "https://example.com/course"

        result = search_tool._format_results(search_results)

        assert "[Test Course]" in result
        assert "Lesson" not in result
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]['link'] == "https://example.com/course"


class TestCourseOutlineTool:
    """Test cases for CourseOutlineTool"""

    @pytest.fixture
    def outline_tool(self, mock_vector_store):
        """Create CourseOutlineTool with mocked vector store"""
        return CourseOutlineTool(mock_vector_store)

    @pytest.fixture
    def course_metadata(self):
        """Sample course metadata"""
        return {
            "title": "Test Course",
            "instructor": "Test Instructor",
            "course_link": "https://example.com/course",
            "lessons": [
                {
                    "lesson_number": 1,
                    "lesson_title": "Introduction",
                    "lesson_link": "https://example.com/lesson1"
                },
                {
                    "lesson_number": 2,
                    "lesson_title": "Advanced Topics",
                    "lesson_link": "https://example.com/lesson2"
                }
            ]
        }

    def test_get_tool_definition(self, outline_tool):
        """Test that outline tool definition is correctly structured"""
        definition = outline_tool.get_tool_definition()

        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "course_name" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["course_name"]

    def test_execute_successful_outline(self, outline_tool, mock_vector_store, course_metadata):
        """Test successful course outline retrieval"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_vector_store.get_all_courses_metadata.return_value = [course_metadata]

        result = outline_tool.execute("Test")

        assert "**Test Course** (by Test Instructor)" in result
        assert "Course Link: https://example.com/course" in result
        assert "Course Outline (2 lessons):" in result
        assert "1. Introduction" in result
        assert "2. Advanced Topics" in result

    def test_execute_course_not_found(self, outline_tool, mock_vector_store):
        """Test when course name cannot be resolved"""
        mock_vector_store._resolve_course_name.return_value = None

        result = outline_tool.execute("NonExistent")

        assert "No course found matching 'NonExistent'" in result

    def test_execute_metadata_not_found(self, outline_tool, mock_vector_store):
        """Test when resolved course metadata is not found"""
        mock_vector_store._resolve_course_name.return_value = "Resolved Course"
        mock_vector_store.get_all_courses_metadata.return_value = []

        result = outline_tool.execute("Test")

        assert "Course metadata not found for 'Resolved Course'" in result

    def test_format_course_outline_minimal(self, outline_tool):
        """Test formatting with minimal course metadata"""
        minimal_metadata = {
            "title": "Minimal Course"
        }

        result = outline_tool._format_course_outline(minimal_metadata)

        assert "**Minimal Course**" in result
        assert "No lesson details available" in result

    def test_format_course_outline_complete(self, outline_tool, course_metadata):
        """Test formatting with complete course metadata"""
        result = outline_tool._format_course_outline(course_metadata)

        assert "**Test Course** (by Test Instructor)" in result
        assert "Course Link: https://example.com/course" in result
        assert "Course Outline (2 lessons):" in result
        assert "1. Introduction" in result
        assert "2. Advanced Topics" in result


class TestToolManager:
    """Test cases for ToolManager"""

    @pytest.fixture
    def tool_manager(self):
        """Create empty ToolManager"""
        return ToolManager()

    @pytest.fixture
    def populated_tool_manager(self, mock_vector_store):
        """Create ToolManager with registered tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        return manager

    def test_register_tool_success(self, tool_manager, mock_vector_store):
        """Test successful tool registration"""
        search_tool = CourseSearchTool(mock_vector_store)

        tool_manager.register_tool(search_tool)

        assert "search_course_content" in tool_manager.tools
        assert tool_manager.tools["search_course_content"] == search_tool

    def test_register_tool_without_name(self, tool_manager):
        """Test tool registration fails without name in definition"""
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {}  # Missing 'name'

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            tool_manager.register_tool(mock_tool)

    def test_get_tool_definitions(self, populated_tool_manager):
        """Test getting all tool definitions"""
        definitions = populated_tool_manager.get_tool_definitions()

        assert len(definitions) == 2
        tool_names = [d["name"] for d in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_execute_tool_success(self, populated_tool_manager, mock_vector_store, search_results_success):
        """Test successful tool execution"""
        mock_vector_store.search.return_value = search_results_success

        result = populated_tool_manager.execute_tool("search_course_content", query="test")

        assert "Software Testing Course" in result

    def test_execute_tool_not_found(self, populated_tool_manager):
        """Test execution of non-existent tool"""
        result = populated_tool_manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, populated_tool_manager, mock_vector_store, search_results_success):
        """Test getting sources from last search operation"""
        mock_vector_store.search.return_value = search_results_success

        # Execute a search to populate sources
        populated_tool_manager.execute_tool("search_course_content", query="test")

        sources = populated_tool_manager.get_last_sources()
        assert len(sources) > 0

    def test_reset_sources(self, populated_tool_manager, mock_vector_store, search_results_success):
        """Test resetting sources after operations"""
        mock_vector_store.search.return_value = search_results_success

        # Execute a search to populate sources
        populated_tool_manager.execute_tool("search_course_content", query="test")
        assert len(populated_tool_manager.get_last_sources()) > 0

        # Reset sources
        populated_tool_manager.reset_sources()
        assert len(populated_tool_manager.get_last_sources()) == 0