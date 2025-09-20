import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from rag_system import RAGSystem
from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Integration tests for RAGSystem"""

    @pytest.fixture
    def mock_components(self):
        """Mock all RAG system components"""
        mocks = {}

        # Mock DocumentProcessor
        with patch('rag_system.DocumentProcessor') as mock_dp:
            mock_instance = Mock()
            mock_dp.return_value = mock_instance
            mocks['document_processor'] = mock_instance

        # Mock VectorStore
        with patch('rag_system.VectorStore') as mock_vs:
            mock_instance = Mock()
            mock_vs.return_value = mock_instance
            mocks['vector_store'] = mock_instance

        # Mock AIGenerator
        with patch('rag_system.AIGenerator') as mock_ai:
            mock_instance = Mock()
            mock_ai.return_value = mock_instance
            mocks['ai_generator'] = mock_instance

        # Mock SessionManager
        with patch('rag_system.SessionManager') as mock_sm:
            mock_instance = Mock()
            mock_sm.return_value = mock_instance
            mocks['session_manager'] = mock_instance

        # Mock ToolManager
        with patch('rag_system.ToolManager') as mock_tm:
            mock_instance = Mock()
            mock_tm.return_value = mock_instance
            mocks['tool_manager'] = mock_instance

        # Mock CourseSearchTool
        with patch('rag_system.CourseSearchTool') as mock_st:
            mock_instance = Mock()
            mock_st.return_value = mock_instance
            mocks['search_tool'] = mock_instance

        # Mock CourseOutlineTool
        with patch('rag_system.CourseOutlineTool') as mock_ot:
            mock_instance = Mock()
            mock_ot.return_value = mock_instance
            mocks['outline_tool'] = mock_instance

        return mocks

    @pytest.fixture
    def rag_system(self, mock_config, mock_components):
        """Create RAGSystem with mocked components"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'), \
             patch('rag_system.CourseOutlineTool'):

            return RAGSystem(mock_config)

    def test_initialization(self, rag_system, mock_config):
        """Test RAGSystem initialization"""
        assert rag_system.config == mock_config
        assert hasattr(rag_system, 'document_processor')
        assert hasattr(rag_system, 'vector_store')
        assert hasattr(rag_system, 'ai_generator')
        assert hasattr(rag_system, 'session_manager')
        assert hasattr(rag_system, 'tool_manager')

    def test_add_course_document_success(self, rag_system, sample_course, sample_chunks):
        """Test successful addition of a course document"""
        # Mock document processor to return sample course and chunks
        rag_system.document_processor.process_course_document.return_value = (sample_course, sample_chunks)

        course, chunk_count = rag_system.add_course_document("/path/to/course.pdf")

        assert course == sample_course
        assert chunk_count == len(sample_chunks)

        # Verify methods were called
        rag_system.document_processor.process_course_document.assert_called_once_with("/path/to/course.pdf")
        rag_system.vector_store.add_course_metadata.assert_called_once_with(sample_course)
        rag_system.vector_store.add_course_content.assert_called_once_with(sample_chunks)

    def test_add_course_document_failure(self, rag_system):
        """Test handling of document processing failure"""
        # Mock document processor to raise exception
        rag_system.document_processor.process_course_document.side_effect = Exception("Processing error")

        with patch('builtins.print') as mock_print:
            course, chunk_count = rag_system.add_course_document("/invalid/path.pdf")

        assert course is None
        assert chunk_count == 0
        mock_print.assert_called_once()

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_success(self, mock_listdir, mock_exists, rag_system, sample_course, sample_chunks):
        """Test successful addition of course folder"""
        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ['course1.pdf', 'course2.docx', 'readme.txt']

        # Mock vector store methods
        rag_system.vector_store.get_existing_course_titles.return_value = []

        # Mock document processor
        rag_system.document_processor.process_course_document.return_value = (sample_course, sample_chunks)

        total_courses, total_chunks = rag_system.add_course_folder("/docs/folder")

        assert total_courses == 2  # PDF and DOCX files
        assert total_chunks == 2 * len(sample_chunks)

        # Verify clear_all_data was not called (default behavior)
        rag_system.vector_store.clear_all_data.assert_not_called()

    @patch('os.path.exists')
    def test_add_course_folder_not_exists(self, mock_exists, rag_system):
        """Test adding course folder that doesn't exist"""
        mock_exists.return_value = False

        with patch('builtins.print') as mock_print:
            total_courses, total_chunks = rag_system.add_course_folder("/nonexistent/folder")

        assert total_courses == 0
        assert total_chunks == 0
        mock_print.assert_called_once()

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_clear_existing(self, mock_listdir, mock_exists, rag_system, sample_course, sample_chunks):
        """Test adding course folder with clear_existing=True"""
        mock_exists.return_value = True
        mock_listdir.return_value = ['course1.pdf']

        rag_system.vector_store.get_existing_course_titles.return_value = []
        rag_system.document_processor.process_course_document.return_value = (sample_course, sample_chunks)

        with patch('builtins.print'):
            rag_system.add_course_folder("/docs/folder", clear_existing=True)

        # Verify clear_all_data was called
        rag_system.vector_store.clear_all_data.assert_called_once()

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_skip_existing(self, mock_listdir, mock_exists, rag_system, sample_course, sample_chunks):
        """Test skipping courses that already exist"""
        mock_exists.return_value = True
        mock_listdir.return_value = ['course1.pdf']

        # Mock that course already exists
        rag_system.vector_store.get_existing_course_titles.return_value = [sample_course.title]
        rag_system.document_processor.process_course_document.return_value = (sample_course, sample_chunks)

        with patch('builtins.print') as mock_print:
            total_courses, total_chunks = rag_system.add_course_folder("/docs/folder")

        assert total_courses == 0  # Should skip existing course
        assert total_chunks == 0
        mock_print.assert_called()  # Should print skip message

    def test_query_success_no_session(self, rag_system):
        """Test successful query without session"""
        # Mock AI generator response
        rag_system.ai_generator.generate_response.return_value = "This is the answer"

        # Mock tool manager sources
        mock_sources = [
            {'text': 'Course 1 - Lesson 1', 'link': 'http://example.com/lesson1'},
            {'text': 'Course 1 - Lesson 2', 'link': 'http://example.com/lesson2'}
        ]
        rag_system.tool_manager.get_last_sources.return_value = mock_sources

        response, sources = rag_system.query("What is Python?")

        assert response == "This is the answer"
        assert sources == mock_sources

        # Verify AI generator was called with correct parameters
        rag_system.ai_generator.generate_response.assert_called_once()
        call_args = rag_system.ai_generator.generate_response.call_args

        assert "Answer this question about course materials: What is Python?" in call_args[1]['query']
        assert call_args[1]['conversation_history'] is None
        assert call_args[1]['tools'] is not None
        assert call_args[1]['tool_manager'] is not None

        # Verify sources were reset
        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_with_session(self, rag_system):
        """Test query with session management"""
        session_id = "test_session_123"
        mock_history = "Previous conversation context"

        # Mock session manager
        rag_system.session_manager.get_conversation_history.return_value = mock_history

        # Mock AI generator response
        rag_system.ai_generator.generate_response.return_value = "Session-aware response"

        # Mock sources
        rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = rag_system.query("Follow up question", session_id=session_id)

        assert response == "Session-aware response"

        # Verify session methods were called
        rag_system.session_manager.get_conversation_history.assert_called_once_with(session_id)
        rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow up question", "Session-aware response"
        )

        # Verify AI generator received conversation history
        call_args = rag_system.ai_generator.generate_response.call_args
        assert call_args[1]['conversation_history'] == mock_history

    def test_query_with_course_filter(self, rag_system):
        """Test query with course filter"""
        course_filter = "Python Programming Course"

        rag_system.ai_generator.generate_response.return_value = "Filtered response"
        rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = rag_system.query("What is a function?", course_filter=course_filter)

        # Verify course filter was included in the prompt
        call_args = rag_system.ai_generator.generate_response.call_args
        query_text = call_args[1]['query']
        assert f"Focus your search on the course: '{course_filter}'" in query_text

    def test_query_ai_generator_exception(self, rag_system):
        """Test query when AI generator raises exception"""
        rag_system.ai_generator.generate_response.side_effect = Exception("AI generation failed")

        # This should propagate the exception (not caught by RAGSystem)
        with pytest.raises(Exception, match="AI generation failed"):
            rag_system.query("Test query")

    def test_get_course_analytics(self, rag_system):
        """Test getting course analytics"""
        # Mock vector store analytics methods
        rag_system.vector_store.get_course_count.return_value = 5
        rag_system.vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]

        analytics = rag_system.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]

    def test_tool_manager_setup(self, rag_system):
        """Test that tool manager is properly set up"""
        # Verify tools were registered
        rag_system.tool_manager.register_tool.assert_any_call(rag_system.search_tool)
        rag_system.tool_manager.register_tool.assert_any_call(rag_system.outline_tool)

    def test_query_prompt_construction(self, rag_system):
        """Test that query prompts are constructed correctly"""
        rag_system.ai_generator.generate_response.return_value = "Test response"
        rag_system.tool_manager.get_last_sources.return_value = []

        # Test basic query
        rag_system.query("What is AI?")
        call_args = rag_system.ai_generator.generate_response.call_args
        assert "Answer this question about course materials: What is AI?" in call_args[1]['query']

        # Test query with course filter
        rag_system.query("What is AI?", course_filter="AI Course")
        call_args = rag_system.ai_generator.generate_response.call_args
        query_text = call_args[1]['query']
        assert "Answer this question about course materials: What is AI?" in query_text
        assert "Focus your search on the course: 'AI Course'" in query_text

    def test_query_tool_definitions_passed(self, rag_system):
        """Test that tool definitions are passed to AI generator"""
        mock_tool_definitions = [
            {"name": "search_course_content", "description": "Search tool"},
            {"name": "get_course_outline", "description": "Outline tool"}
        ]

        rag_system.tool_manager.get_tool_definitions.return_value = mock_tool_definitions
        rag_system.ai_generator.generate_response.return_value = "Tool-enabled response"
        rag_system.tool_manager.get_last_sources.return_value = []

        rag_system.query("Test query")

        call_args = rag_system.ai_generator.generate_response.call_args
        assert call_args[1]['tools'] == mock_tool_definitions
        assert call_args[1]['tool_manager'] == rag_system.tool_manager

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_processing_error(self, mock_listdir, mock_exists, rag_system):
        """Test handling of processing errors in course folder"""
        mock_exists.return_value = True
        mock_listdir.return_value = ['course1.pdf', 'course2.pdf']

        rag_system.vector_store.get_existing_course_titles.return_value = []

        # First call succeeds, second fails
        rag_system.document_processor.process_course_document.side_effect = [
            (Mock(), [Mock()]),  # Success
            Exception("Processing error")  # Failure
        ]

        with patch('builtins.print') as mock_print:
            total_courses, total_chunks = rag_system.add_course_folder("/docs/folder")

        # Should have processed one course successfully, one failed
        assert total_courses == 1
        assert total_chunks == 1
        mock_print.assert_called()  # Should print error message

    def test_query_sources_handling(self, rag_system):
        """Test that sources are properly handled and reset"""
        mock_sources = [{'text': 'Test Source', 'link': 'http://example.com'}]

        rag_system.ai_generator.generate_response.return_value = "Response with sources"
        rag_system.tool_manager.get_last_sources.return_value = mock_sources

        response, sources = rag_system.query("Test query")

        assert sources == mock_sources

        # Verify sources were retrieved and reset
        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()

    def test_session_handling_edge_cases(self, rag_system):
        """Test session handling edge cases"""
        # Test with empty session ID
        rag_system.ai_generator.generate_response.return_value = "Response"
        rag_system.tool_manager.get_last_sources.return_value = []

        response, sources = rag_system.query("Test", session_id="")

        # Should treat empty string as no session
        rag_system.session_manager.get_conversation_history.assert_not_called()
        rag_system.session_manager.add_exchange.assert_not_called()

        # Test with None session ID explicitly
        response, sources = rag_system.query("Test", session_id=None)

        rag_system.session_manager.get_conversation_history.assert_not_called()
        rag_system.session_manager.add_exchange.assert_not_called()