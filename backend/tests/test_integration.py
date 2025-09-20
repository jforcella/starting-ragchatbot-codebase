import pytest
import sys
import os
from unittest.mock import patch, Mock

# Add backend to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import config
from rag_system import RAGSystem
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool


class TestLiveIntegration:
    """Integration tests with actual components (but mocked external services)"""

    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        test_config = config.__class__()
        test_config.ANTHROPIC_API_KEY = "test-key-123"
        test_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        test_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        test_config.CHUNK_SIZE = 800
        test_config.CHUNK_OVERLAP = 100
        test_config.MAX_RESULTS = 5
        test_config.MAX_HISTORY = 2
        test_config.CHROMA_PATH = "./test_integration_chroma_db"
        return test_config

    def test_course_search_tool_with_empty_vector_store(self, test_config):
        """Test CourseSearchTool behavior with empty vector store"""
        # Create a real vector store (but empty)
        with patch('chromadb.PersistentClient') as mock_client:
            # Mock empty ChromaDB responses
            mock_collection = Mock()

            # Mock empty search results
            mock_collection.query.return_value = {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }

            mock_client.return_value.get_or_create_collection.return_value = mock_collection

            # Create actual VectorStore
            vector_store = VectorStore(
                chroma_path=test_config.CHROMA_PATH,
                embedding_model=test_config.EMBEDDING_MODEL,
                max_results=test_config.MAX_RESULTS
            )

            # Create actual CourseSearchTool
            search_tool = CourseSearchTool(vector_store)

            # Test execution with empty store
            result = search_tool.execute("What is Python programming?")

            # Should return "No relevant content found"
            assert "No relevant content found" in result
            assert isinstance(result, str)
            print(f"Empty store result: {result}")

    def test_course_search_tool_with_chromadb_error(self, test_config):
        """Test CourseSearchTool behavior when ChromaDB raises exceptions"""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = Mock()

            # Mock ChromaDB to raise exception
            mock_collection.query.side_effect = Exception("ChromaDB connection failed")

            mock_client.return_value.get_or_create_collection.return_value = mock_collection

            # Create actual VectorStore
            vector_store = VectorStore(
                chroma_path=test_config.CHROMA_PATH,
                embedding_model=test_config.EMBEDDING_MODEL,
                max_results=test_config.MAX_RESULTS
            )

            # Create actual CourseSearchTool
            search_tool = CourseSearchTool(vector_store)

            # Test execution with error
            result = search_tool.execute("What is Python programming?")

            # Should return error message
            assert "Search error: ChromaDB connection failed" in result
            print(f"ChromaDB error result: {result}")

    def test_course_search_tool_with_nonexistent_course(self, test_config):
        """Test CourseSearchTool behavior when course resolution fails"""
        with patch('chromadb.PersistentClient') as mock_client:
            mock_collection = Mock()

            # Mock course catalog to return no results (course not found)
            catalog_collection = Mock()
            catalog_collection.query.return_value = {
                'documents': [[]],
                'metadatas': [[]]
            }

            content_collection = Mock()

            # Return different collections for catalog vs content
            def get_collection_side_effect(name, **kwargs):
                if name == "course_catalog":
                    return catalog_collection
                elif name == "course_content":
                    return content_collection
                return mock_collection

            mock_client.return_value.get_or_create_collection.side_effect = get_collection_side_effect

            # Create actual VectorStore
            vector_store = VectorStore(
                chroma_path=test_config.CHROMA_PATH,
                embedding_model=test_config.EMBEDDING_MODEL,
                max_results=test_config.MAX_RESULTS
            )

            # Create actual CourseSearchTool
            search_tool = CourseSearchTool(vector_store)

            # Test execution with course filter that doesn't exist
            result = search_tool.execute("What is Python?", course_name="NonExistent Course")

            # Should return course not found error
            assert "No course found matching 'NonExistent Course'" in result
            print(f"Course not found result: {result}")

    def test_rag_system_query_with_tool_execution_failure(self, test_config):
        """Test RAG system when tool execution fails"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs_class, \
             patch('rag_system.AIGenerator') as mock_ai_class, \
             patch('rag_system.SessionManager'), \
             patch('chromadb.PersistentClient'):

            # Mock VectorStore to return search errors
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = SearchResults.empty("Mock search error")
            mock_vs_class.return_value = mock_vector_store

            # Mock AIGenerator for tool execution
            mock_ai_generator = Mock()

            # Create mock tool use response
            mock_tool_block = Mock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.input = {"query": "test query"}
            mock_tool_block.id = "tool_123"

            mock_initial_response = Mock()
            mock_initial_response.stop_reason = "tool_use"
            mock_initial_response.content = [mock_tool_block]

            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="I couldn't find relevant information about your query.")]

            # Mock AI to return tool use, then final response
            mock_ai_generator.generate_response.return_value = "I couldn't find relevant information about your query."
            mock_ai_class.return_value = mock_ai_generator

            # Create RAG system
            rag_system = RAGSystem(test_config)

            # Test query that should trigger tool execution
            response, sources = rag_system.query("What is Python programming?")

            # Should get a response even if search fails
            assert isinstance(response, str)
            assert len(response) > 0
            print(f"RAG system response with search failure: {response}")

    def test_search_with_valid_data(self, test_config):
        """Test search with some valid mock data"""
        with patch('chromadb.PersistentClient') as mock_client:
            # Mock collections with valid data
            mock_content_collection = Mock()
            mock_catalog_collection = Mock()

            # Mock search results with valid data
            mock_content_collection.query.return_value = {
                'documents': [['Python is a programming language used for web development, data science, and automation.']],
                'metadatas': [[{
                    'course_title': 'Python Programming Course',
                    'lesson_number': 1,
                    'chunk_index': 0
                }]],
                'distances': [[0.2]]
            }

            def get_collection_side_effect(name, **kwargs):
                if name == "course_catalog":
                    return mock_catalog_collection
                elif name == "course_content":
                    return mock_content_collection
                return Mock()

            mock_client.return_value.get_or_create_collection.side_effect = get_collection_side_effect

            # Create actual VectorStore
            vector_store = VectorStore(
                chroma_path=test_config.CHROMA_PATH,
                embedding_model=test_config.EMBEDDING_MODEL,
                max_results=test_config.MAX_RESULTS
            )

            # Create actual CourseSearchTool
            search_tool = CourseSearchTool(vector_store)

            # Test execution with valid data
            result = search_tool.execute("What is Python?")

            # Should return formatted results
            assert "Python Programming Course" in result
            assert "Python is a programming language" in result
            print(f"Valid data search result: {result}")

if __name__ == "__main__":
    # Run a quick test to see what happens
    test = TestLiveIntegration()

    # Test configuration
    test_config = config.__class__()
    test_config.ANTHROPIC_API_KEY = "test-key-123"
    test_config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    test_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    test_config.CHUNK_SIZE = 800
    test_config.CHUNK_OVERLAP = 100
    test_config.MAX_RESULTS = 5
    test_config.MAX_HISTORY = 2
    test_config.CHROMA_PATH = "./test_integration_chroma_db"

    try:
        test.test_course_search_tool_with_empty_vector_store(test_config)
        print("✓ Empty vector store test passed")

        test.test_course_search_tool_with_chromadb_error(test_config)
        print("✓ ChromaDB error test passed")

        test.test_course_search_tool_with_nonexistent_course(test_config)
        print("✓ Nonexistent course test passed")

        test.test_search_with_valid_data(test_config)
        print("✓ Valid data search test passed")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()