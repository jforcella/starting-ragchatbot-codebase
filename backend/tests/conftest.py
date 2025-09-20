import pytest
import sys
import os
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(
            lesson_number=1,
            title="Introduction to Testing",
            lesson_link="https://example.com/lesson1"
        ),
        Lesson(
            lesson_number=2,
            title="Advanced Testing Techniques",
            lesson_link="https://example.com/lesson2"
        )
    ]

    return Course(
        title="Software Testing Course",
        instructor="Test Instructor",
        course_link="https://example.com/course",
        lessons=lessons
    )


@pytest.fixture
def sample_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=1,
            content="This is lesson 1 content about introduction to testing.",
            chunk_index=0
        ),
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=1,
            content="More content from lesson 1 about testing basics.",
            chunk_index=1
        ),
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=2,
            content="This is lesson 2 content about advanced techniques.",
            chunk_index=2
        )
    ]


@pytest.fixture
def search_results_success():
    """Create successful search results for testing"""
    return SearchResults(
        documents=[
            "This is lesson 1 content about introduction to testing.",
            "This is lesson 2 content about advanced techniques."
        ],
        metadata=[
            {
                "course_title": "Software Testing Course",
                "lesson_number": 1,
                "chunk_index": 0
            },
            {
                "course_title": "Software Testing Course",
                "lesson_number": 2,
                "chunk_index": 2
            }
        ],
        distances=[0.2, 0.3]
    )


@pytest.fixture
def search_results_empty():
    """Create empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def search_results_error():
    """Create error search results for testing"""
    return SearchResults.empty("Test error: ChromaDB connection failed")


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()

    # Mock successful response without tool use
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a test response")]

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic client response with tool use"""
    mock_client = Mock()

    # Mock tool use response
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.input = {"query": "test query"}
    mock_tool_block.id = "tool_123"

    mock_initial_response = Mock()
    mock_initial_response.stop_reason = "tool_use"
    mock_initial_response.content = [mock_tool_block]

    # Mock final response after tool execution
    mock_final_response = Mock()
    mock_final_response.content = [Mock(text="Final response after tool use")]

    # Set up the mock to return initial response first, then final response
    mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]

    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing"""
    mock_store = Mock()

    # Default successful search
    mock_store.search.return_value = SearchResults(
        documents=["Test content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.2]
    )

    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    mock_store.get_course_link.return_value = "https://example.com/course"

    return mock_store


# Cleanup fixture to remove test database files
@pytest.fixture(autouse=True)
def cleanup_test_db():
    """Automatically cleanup test database files after each test"""
    yield
    # Cleanup any test database files if they exist
    import shutil
    test_db_path = "./test_chroma_db"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path, ignore_errors=True)