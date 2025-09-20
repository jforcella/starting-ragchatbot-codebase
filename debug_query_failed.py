#!/usr/bin/env python3
"""
Debug script to identify the root cause of 'query failed' errors
"""

import sys
import os
sys.path.append('./backend')

from config import config
from rag_system import RAGSystem
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool
import traceback


def test_vector_store_directly():
    """Test VectorStore directly to see if it's working"""
    print("=== Testing VectorStore Directly ===")

    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)

        # Test basic search
        print(f"1. Testing basic search...")
        results = vector_store.search("Python programming")
        print(f"   Results error: {results.error}")
        print(f"   Results empty: {results.is_empty()}")
        print(f"   Documents count: {len(results.documents)}")

        # Test course count
        print(f"2. Testing course count...")
        course_count = vector_store.get_course_count()
        print(f"   Course count: {course_count}")

        # Test existing course titles
        print(f"3. Testing existing course titles...")
        course_titles = vector_store.get_existing_course_titles()
        print(f"   Course titles: {course_titles}")

        # Test course resolution
        print(f"4. Testing course name resolution...")
        if course_titles:
            resolved = vector_store._resolve_course_name(course_titles[0])
            print(f"   Resolved '{course_titles[0]}' to: {resolved}")

        return True

    except Exception as e:
        print(f"   VectorStore test failed: {e}")
        traceback.print_exc()
        return False


def test_search_tool_directly():
    """Test CourseSearchTool directly"""
    print("\n=== Testing CourseSearchTool Directly ===")

    try:
        vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        search_tool = CourseSearchTool(vector_store)

        # Test basic search
        print("1. Testing basic search...")
        result = search_tool.execute("What is programming?")
        print(f"   Search result: {result[:200]}...")

        # Test with course filter
        course_titles = vector_store.get_existing_course_titles()
        if course_titles:
            print("2. Testing search with course filter...")
            result = search_tool.execute("What is programming?", course_name=course_titles[0])
            print(f"   Filtered search result: {result[:200]}...")

        # Test with invalid course
        print("3. Testing search with invalid course...")
        result = search_tool.execute("What is programming?", course_name="NonExistent Course")
        print(f"   Invalid course result: {result}")

        return True

    except Exception as e:
        print(f"   CourseSearchTool test failed: {e}")
        traceback.print_exc()
        return False


def test_rag_system_components():
    """Test RAG system component by component"""
    print("\n=== Testing RAG System Components ===")

    try:
        # Initialize RAG system
        print("1. Initializing RAG system...")
        rag_system = RAGSystem(config)
        print("   ✓ RAG system initialized")

        # Test tool manager
        print("2. Testing tool manager...")
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        print(f"   Available tools: {[tool['name'] for tool in tool_definitions]}")

        # Test tool execution directly
        print("3. Testing tool execution...")
        tool_result = rag_system.tool_manager.execute_tool(
            "search_course_content",
            query="What is programming?"
        )
        print(f"   Tool result: {tool_result[:200]}...")

        # Test sources
        print("4. Testing sources...")
        sources = rag_system.tool_manager.get_last_sources()
        print(f"   Sources count: {len(sources)}")
        if sources:
            print(f"   First source: {sources[0]}")

        return True

    except Exception as e:
        print(f"   RAG system component test failed: {e}")
        traceback.print_exc()
        return False


def test_full_rag_query():
    """Test full RAG system query (without AI generation)"""
    print("\n=== Testing Full RAG Query (Mock AI) ===")

    try:
        # Initialize RAG system
        rag_system = RAGSystem(config)

        # Mock the AI generator to avoid API calls
        original_generate = rag_system.ai_generator.generate_response

        def mock_generate_response(query, conversation_history=None, tools=None, tool_manager=None):
            print(f"   AI Generator called with query: {query[:100]}...")
            if tools and tool_manager:
                # Simulate tool use
                print("   Simulating tool execution...")
                tool_result = tool_manager.execute_tool("search_course_content", query="test query")
                return f"Based on the search results: {tool_result[:200]}..."
            return "Mock AI response without tools"

        rag_system.ai_generator.generate_response = mock_generate_response

        # Test query
        print("1. Testing RAG query...")
        response, sources = rag_system.query("What is Python programming?")
        print(f"   Response: {response[:300]}...")
        print(f"   Sources count: {len(sources)}")

        return True

    except Exception as e:
        print(f"   Full RAG query test failed: {e}")
        traceback.print_exc()
        return False


def diagnose_chromadb_state():
    """Check the state of ChromaDB directly"""
    print("\n=== Diagnosing ChromaDB State ===")

    try:
        import chromadb
        from chromadb.config import Settings

        # Connect directly to ChromaDB
        client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
        )

        # List collections
        collections = client.list_collections()
        print(f"1. Collections found: {[c.name for c in collections]}")

        # Check each collection
        for collection in collections:
            count = collection.count()
            print(f"   Collection '{collection.name}': {count} items")

            if count > 0:
                # Get a few samples
                sample = collection.peek(limit=3)
                print(f"   Sample documents: {len(sample.get('documents', []))}")
                print(f"   Sample metadata: {len(sample.get('metadatas', []))}")

                if sample.get('documents'):
                    print(f"   First document preview: {sample['documents'][0][:100]}...")

        return True

    except Exception as e:
        print(f"   ChromaDB diagnosis failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests"""
    print("🔍 Diagnosing RAG System for 'Query Failed' Issues")
    print("=" * 60)

    # Check if chromadb exists
    if not os.path.exists(config.CHROMA_PATH):
        print(f"❌ ChromaDB path doesn't exist: {config.CHROMA_PATH}")
        print("   This might be why queries are failing!")
        return

    results = []

    # Run all tests
    results.append(("ChromaDB State", diagnose_chromadb_state()))
    results.append(("VectorStore", test_vector_store_directly()))
    results.append(("CourseSearchTool", test_search_tool_directly()))
    results.append(("RAG Components", test_rag_system_components()))
    results.append(("Full RAG Query", test_full_rag_query()))

    # Summary
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSIS SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {test_name}")

    failed_tests = [name for name, passed in results if not passed]

    if failed_tests:
        print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        print("These components are likely causing 'query failed' errors!")
    else:
        print("\n✅ All tests passed - the issue might be elsewhere or intermittent")


if __name__ == "__main__":
    main()