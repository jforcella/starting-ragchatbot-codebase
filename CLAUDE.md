# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Application
- **Main command**: `./run.sh` (preferred) - starts the RAG system with automatic document loading
- **Manual start**: `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Dependencies**: `uv sync` - installs all Python dependencies via uv package manager

### Environment Setup
- Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
- Requires Python 3.13+ and uv package manager

### Application URLs
- Web interface: http://localhost:8000
- API documentation: http://localhost:8000/docs

### Testing & Validation
- **Health check**: Visit http://localhost:8000/docs to verify API is running
- **Test query**: Use the web interface to ask a sample question
- **Dependency management**: `uv sync` to sync dependencies, `uv add <package>` to add new ones
- **Reset database**: Delete `chroma_db/` directory to clear all processed documents and start fresh

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for querying course materials using semantic search and Claude AI.

### Backend Architecture (`/backend`)
- **`app.py`** - FastAPI application with CORS, endpoints for queries (`/api/query`) and course stats (`/api/courses`)
- **`rag_system.py`** - Main orchestrator coordinating all components
- **`config.py`** - Configuration management with dataclass pattern, loads from .env
- **`vector_store.py`** - ChromaDB integration for semantic search and embeddings
- **`document_processor.py`** - Text chunking and course metadata extraction
- **`ai_generator.py`** - Anthropic Claude API integration with tool support
- **`session_manager.py`** - Conversation history management
- **`search_tools.py`** - Tool-based search system for AI agent
- **`models.py`** - Data models (Course, Lesson, CourseChunk)

### Root Level
- **`main.py`** - Entry point (if using alternative to run.sh)
- **`run.sh`** - Main startup script with directory setup
- **`pyproject.toml`** - Python dependencies and project configuration

### Key Components
- **Vector Database**: ChromaDB with sentence-transformers embeddings (all-MiniLM-L6-v2)
- **AI Model**: Claude Sonnet 4 (claude-sonnet-4-20250514)
- **Text Chunking**: 800 chars with 100 char overlap
- **Document Processing**: Supports PDF, DOCX, TXT files

### Frontend (`/frontend`)
- Static HTML/CSS/JS served by FastAPI
- Web interface for querying and displaying course statistics
- Auto-loads documents from `/docs` directory on startup

### Data Flow
1. Documents in `/docs` are automatically processed on startup
2. Text is chunked and stored in ChromaDB with embeddings
3. User queries trigger semantic search via AI tools
4. Claude generates responses using retrieved context
5. Session management maintains conversation history

### Configuration
- **Chunk size**: 800 characters (configurable in config.py:19)
- **Max results**: 5 search results (config.py:21)  
- **Max history**: 2 conversation turns (config.py:22)
- **ChromaDB path**: `./chroma_db` (config.py:25)

## Development Notes

### Document Processing
- Place documents in `/docs` directory - supports PDF, DOCX, TXT formats
- Documents are auto-processed on startup and stored in ChromaDB
- Delete `chroma_db/` directory to reset the vector database

### Common Issues
- **Missing API key**: Ensure `ANTHROPIC_API_KEY` is set in `.env` file
- **Port conflicts**: Default port 8000 - change in uvicorn command if needed
- **Python version**: Requires Python 3.13+ as specified in pyproject.toml