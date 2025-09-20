import pytest
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
import json

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test cases for SearchResults class"""

    def test_from_chroma_success(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key1': 'val1'}, {'key2': 'val2'}]],
            'distances': [[0.1, 0.2]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key1': 'val1'}, {'key2': 'val2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        error_msg = "Test error message"
        results = SearchResults.empty(error_msg)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg

    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults([], [], [])
        assert results.is_empty() is True

    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(['doc'], [{}], [0.1])
        assert results.is_empty() is False


class TestVectorStore:
    """Test cases for VectorStore class"""

    @pytest.fixture
    def test_vector_store(self, mock_config):
        """Create VectorStore with test configuration"""
        with patch('chromadb.PersistentClient') as mock_client, \
             patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            # Mock ChromaDB client and collections
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection

            vector_store = VectorStore(
                chroma_path=mock_config.CHROMA_PATH,
                embedding_model=mock_config.EMBEDDING_MODEL,
                max_results=mock_config.MAX_RESULTS
            )

            # Store mocks for later use in tests
            vector_store._mock_client = mock_client.return_value
            vector_store._mock_catalog = mock_collection
            vector_store._mock_content = mock_collection

            return vector_store

    def test_initialization(self, test_vector_store, mock_config):
        """Test VectorStore initialization"""
        assert test_vector_store.max_results == mock_config.MAX_RESULTS
        assert hasattr(test_vector_store, 'course_catalog')
        assert hasattr(test_vector_store, 'course_content')

    def test_search_success_no_filters(self, test_vector_store):
        """Test successful search without filters"""
        # Mock ChromaDB query response
        mock_chroma_results = {
            'documents': [['Test document content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
            'distances': [[0.2]]
        }

        test_vector_store.course_content.query.return_value = mock_chroma_results

        results = test_vector_store.search("test query")

        assert not results.error
        assert len(results.documents) == 1
        assert results.documents[0] == "Test document content"

        # Verify query was called with correct parameters
        test_vector_store.course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,  # default max_results
            where=None
        )

    def test_search_with_course_filter(self, test_vector_store):
        """Test search with course name filter"""
        # Mock course resolution
        test_vector_store._resolve_course_name = Mock(return_value="Resolved Course")

        # Mock ChromaDB query response
        mock_chroma_results = {
            'documents': [['Filtered content']],
            'metadatas': [[{'course_title': 'Resolved Course', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }

        test_vector_store.course_content.query.return_value = mock_chroma_results

        results = test_vector_store.search("test query", course_name="Test Course")

        assert not results.error
        assert len(results.documents) == 1

        # Verify course resolution was called
        test_vector_store._resolve_course_name.assert_called_once_with("Test Course")

        # Verify query was called with course filter
        expected_filter = {"course_title": "Resolved Course"}
        test_vector_store.course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=expected_filter
        )

    def test_search_course_not_found(self, test_vector_store):
        """Test search when course name cannot be resolved"""
        test_vector_store._resolve_course_name = Mock(return_value=None)

        results = test_vector_store.search("test query", course_name="NonExistent")

        assert results.error == "No course found matching 'NonExistent'"
        assert results.is_empty()

    def test_search_with_lesson_filter(self, test_vector_store):
        """Test search with lesson number filter"""
        mock_chroma_results = {
            'documents': [['Lesson content']],
            'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 2}]],
            'distances': [[0.15]]
        }

        test_vector_store.course_content.query.return_value = mock_chroma_results

        results = test_vector_store.search("test query", lesson_number=2)

        expected_filter = {"lesson_number": 2}
        test_vector_store.course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=expected_filter
        )

    def test_search_with_both_filters(self, test_vector_store):
        """Test search with both course and lesson filters"""
        test_vector_store._resolve_course_name = Mock(return_value="Resolved Course")

        mock_chroma_results = {
            'documents': [['Specific content']],
            'metadatas': [[{'course_title': 'Resolved Course', 'lesson_number': 3}]],
            'distances': [[0.05]]
        }

        test_vector_store.course_content.query.return_value = mock_chroma_results

        results = test_vector_store.search("test query", course_name="Test", lesson_number=3)

        expected_filter = {"$and": [
            {"course_title": "Resolved Course"},
            {"lesson_number": 3}
        ]}
        test_vector_store.course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=expected_filter
        )

    def test_search_with_custom_limit(self, test_vector_store):
        """Test search with custom result limit"""
        mock_chroma_results = {
            'documents': [['Content 1', 'Content 2']],
            'metadatas': [[{'course_title': 'Test'}, {'course_title': 'Test'}]],
            'distances': [[0.1, 0.2]]
        }

        test_vector_store.course_content.query.return_value = mock_chroma_results

        results = test_vector_store.search("test query", limit=2)

        test_vector_store.course_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=2,
            where=None
        )

    def test_search_chromadb_exception(self, test_vector_store):
        """Test search handles ChromaDB exceptions"""
        test_vector_store.course_content.query.side_effect = Exception("ChromaDB error")

        results = test_vector_store.search("test query")

        assert results.error == "Search error: ChromaDB error"
        assert results.is_empty()

    def test_resolve_course_name_success(self, test_vector_store):
        """Test successful course name resolution"""
        mock_catalog_results = {
            'documents': [['Course Title']],
            'metadatas': [[{'title': 'Resolved Course Title'}]]
        }

        test_vector_store.course_catalog.query.return_value = mock_catalog_results

        resolved = test_vector_store._resolve_course_name("partial name")

        assert resolved == "Resolved Course Title"
        test_vector_store.course_catalog.query.assert_called_once_with(
            query_texts=["partial name"],
            n_results=1
        )

    def test_resolve_course_name_not_found(self, test_vector_store):
        """Test course name resolution when no match found"""
        mock_catalog_results = {
            'documents': [[]],
            'metadatas': [[]]
        }

        test_vector_store.course_catalog.query.return_value = mock_catalog_results

        resolved = test_vector_store._resolve_course_name("nonexistent")

        assert resolved is None

    def test_resolve_course_name_exception(self, test_vector_store):
        """Test course name resolution handles exceptions"""
        test_vector_store.course_catalog.query.side_effect = Exception("Query error")

        with patch('builtins.print') as mock_print:
            resolved = test_vector_store._resolve_course_name("test")

        assert resolved is None
        mock_print.assert_called_once()

    def test_build_filter_no_filters(self, test_vector_store):
        """Test filter building with no filters"""
        result = test_vector_store._build_filter(None, None)
        assert result is None

    def test_build_filter_course_only(self, test_vector_store):
        """Test filter building with course filter only"""
        result = test_vector_store._build_filter("Test Course", None)
        assert result == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, test_vector_store):
        """Test filter building with lesson filter only"""
        result = test_vector_store._build_filter(None, 5)
        assert result == {"lesson_number": 5}

    def test_build_filter_both(self, test_vector_store):
        """Test filter building with both filters"""
        result = test_vector_store._build_filter("Test Course", 5)
        expected = {"$and": [
            {"course_title": "Test Course"},
            {"lesson_number": 5}
        ]}
        assert result == expected

    def test_add_course_metadata(self, test_vector_store, sample_course):
        """Test adding course metadata"""
        test_vector_store.add_course_metadata(sample_course)

        # Verify add was called on course_catalog
        test_vector_store.course_catalog.add.assert_called_once()

        call_args = test_vector_store.course_catalog.add.call_args
        assert call_args[1]['documents'] == [sample_course.title]
        assert call_args[1]['ids'] == [sample_course.title]

        metadata = call_args[1]['metadatas'][0]
        assert metadata['title'] == sample_course.title
        assert metadata['instructor'] == sample_course.instructor
        assert metadata['course_link'] == sample_course.course_link
        assert 'lessons_json' in metadata

    def test_add_course_content(self, test_vector_store, sample_chunks):
        """Test adding course content chunks"""
        test_vector_store.add_course_content(sample_chunks)

        # Verify add was called on course_content
        test_vector_store.course_content.add.assert_called_once()

        call_args = test_vector_store.course_content.add.call_args
        assert len(call_args[1]['documents']) == 3
        assert len(call_args[1]['metadatas']) == 3
        assert len(call_args[1]['ids']) == 3

    def test_add_course_content_empty(self, test_vector_store):
        """Test adding empty course content list"""
        test_vector_store.add_course_content([])

        # Verify add was not called
        test_vector_store.course_content.add.assert_not_called()

    def test_clear_all_data(self, test_vector_store):
        """Test clearing all data"""
        test_vector_store.clear_all_data()

        # Verify delete_collection was called for both collections
        assert test_vector_store._mock_client.delete_collection.call_count == 2

    def test_clear_all_data_exception(self, test_vector_store):
        """Test clear all data handles exceptions"""
        test_vector_store._mock_client.delete_collection.side_effect = Exception("Delete error")

        with patch('builtins.print') as mock_print:
            test_vector_store.clear_all_data()

        mock_print.assert_called_once()

    def test_get_existing_course_titles(self, test_vector_store):
        """Test getting existing course titles"""
        mock_catalog_get = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }

        test_vector_store.course_catalog.get.return_value = mock_catalog_get

        titles = test_vector_store.get_existing_course_titles()

        assert titles == ['Course 1', 'Course 2', 'Course 3']

    def test_get_existing_course_titles_empty(self, test_vector_store):
        """Test getting course titles when none exist"""
        test_vector_store.course_catalog.get.return_value = {}

        titles = test_vector_store.get_existing_course_titles()

        assert titles == []

    def test_get_existing_course_titles_exception(self, test_vector_store):
        """Test getting course titles handles exceptions"""
        test_vector_store.course_catalog.get.side_effect = Exception("Get error")

        with patch('builtins.print') as mock_print:
            titles = test_vector_store.get_existing_course_titles()

        assert titles == []
        mock_print.assert_called_once()

    def test_get_course_count(self, test_vector_store):
        """Test getting course count"""
        mock_catalog_get = {
            'ids': ['Course 1', 'Course 2']
        }

        test_vector_store.course_catalog.get.return_value = mock_catalog_get

        count = test_vector_store.get_course_count()

        assert count == 2

    def test_get_course_count_exception(self, test_vector_store):
        """Test course count handles exceptions"""
        test_vector_store.course_catalog.get.side_effect = Exception("Count error")

        with patch('builtins.print') as mock_print:
            count = test_vector_store.get_course_count()

        assert count == 0
        mock_print.assert_called_once()

    def test_get_all_courses_metadata(self, test_vector_store):
        """Test getting all courses metadata"""
        lessons_data = [
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "link1"},
            {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "link2"}
        ]

        mock_catalog_get = {
            'metadatas': [{
                'title': 'Test Course',
                'instructor': 'Test Instructor',
                'course_link': 'http://example.com',
                'lessons_json': json.dumps(lessons_data)
            }]
        }

        test_vector_store.course_catalog.get.return_value = mock_catalog_get

        metadata = test_vector_store.get_all_courses_metadata()

        assert len(metadata) == 1
        assert metadata[0]['title'] == 'Test Course'
        assert metadata[0]['lessons'] == lessons_data
        assert 'lessons_json' not in metadata[0]  # Should be removed

    def test_get_course_link(self, test_vector_store):
        """Test getting course link"""
        mock_catalog_get = {
            'metadatas': [{'course_link': 'http://example.com/course'}]
        }

        test_vector_store.course_catalog.get.return_value = mock_catalog_get

        link = test_vector_store.get_course_link("Test Course")

        assert link == 'http://example.com/course'
        test_vector_store.course_catalog.get.assert_called_once_with(ids=["Test Course"])

    def test_get_course_link_not_found(self, test_vector_store):
        """Test getting course link when course not found"""
        test_vector_store.course_catalog.get.return_value = {'metadatas': []}

        link = test_vector_store.get_course_link("NonExistent")

        assert link is None

    def test_get_lesson_link(self, test_vector_store):
        """Test getting lesson link"""
        lessons_data = [
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "http://example.com/lesson1"},
            {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "http://example.com/lesson2"}
        ]

        mock_catalog_get = {
            'metadatas': [{
                'lessons_json': json.dumps(lessons_data)
            }]
        }

        test_vector_store.course_catalog.get.return_value = mock_catalog_get

        link = test_vector_store.get_lesson_link("Test Course", 2)

        assert link == "http://example.com/lesson2"

    def test_get_lesson_link_not_found(self, test_vector_store):
        """Test getting lesson link when lesson not found"""
        lessons_data = [
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "http://example.com/lesson1"}
        ]

        mock_catalog_get = {
            'metadatas': [{
                'lessons_json': json.dumps(lessons_data)
            }]
        }

        test_vector_store.course_catalog.get.return_value = mock_catalog_get

        link = test_vector_store.get_lesson_link("Test Course", 99)

        assert link is None