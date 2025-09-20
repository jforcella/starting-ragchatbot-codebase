import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

from config import config
from rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None
    course_filter: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[dict]
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

# API Endpoints

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
        
        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id, request.course_filter)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Check system health and document loading status"""
    try:
        course_count = rag_system.vector_store.get_course_count()
        course_titles = rag_system.vector_store.get_existing_course_titles()

        # Test a simple search to verify system is working
        test_result = rag_system.tool_manager.execute_tool("search_course_content", query="test")
        search_working = not ("No relevant content found" in test_result and course_count > 0)

        status = "healthy" if course_count > 0 else "no_data"
        if course_count > 0 and not search_working:
            status = "degraded"

        return {
            "status": status,
            "courses_loaded": course_count,
            "course_titles": course_titles[:10],  # Limit to first 10 for readability
            "search_functional": search_working,
            "message": _get_health_message(status, course_count),
            "recommendations": _get_health_recommendations(status, course_count)
        }
    except Exception as e:
        return {
            "status": "error",
            "courses_loaded": 0,
            "course_titles": [],
            "search_functional": False,
            "message": f"Health check failed: {str(e)}",
            "recommendations": ["Check system logs", "Restart the application"]
        }

def _get_health_message(status: str, course_count: int) -> str:
    """Generate appropriate health message"""
    if status == "healthy":
        return f"System is healthy with {course_count} courses loaded and search working"
    elif status == "no_data":
        return "System started but no courses loaded - this will cause 'query failed' errors"
    elif status == "degraded":
        return f"Courses loaded ({course_count}) but search not working properly"
    else:
        return "System error detected"

def _get_health_recommendations(status: str, course_count: int) -> List[str]:
    """Generate health recommendations"""
    if status == "healthy":
        return ["System is operating normally"]
    elif status == "no_data":
        return [
            "Check if documents exist in /docs directory",
            "Verify document loading during startup",
            "Check file permissions on documents",
            "Try restarting the application"
        ]
    elif status == "degraded":
        return [
            "Check ChromaDB connection",
            "Verify vector store integrity",
            "Check system resources",
            "Review application logs"
        ]
    else:
        return ["Check application logs", "Restart the application", "Verify system configuration"]

@app.post("/api/reload-documents")
async def reload_documents():
    """Manually reload documents from the docs directory"""
    try:
        docs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")

        if not os.path.exists(docs_path):
            raise HTTPException(
                status_code=404,
                detail=f"Documents directory not found: {docs_path}"
            )

        # Clear existing data and reload
        print("🔄 Manually reloading documents...")
        courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=True)

        # Verify reload
        total_courses = rag_system.vector_store.get_course_count()

        return {
            "success": True,
            "message": f"Successfully reloaded {courses} courses with {chunks} chunks",
            "courses_loaded": courses,
            "chunks_created": chunks,
            "total_courses_in_store": total_courses,
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }

    except Exception as e:
        print(f"❌ Document reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload documents: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup with better error handling"""
    # Use absolute path resolution to avoid path issues
    docs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")

    if os.path.exists(docs_path):
        print(f"Loading documents from: {docs_path}")
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            print(f"✅ Successfully loaded {courses} courses with {chunks} chunks")

            # Verify loading worked
            total_courses = rag_system.vector_store.get_course_count()
            if total_courses == 0:
                print("⚠️  Warning: No courses found in vector store after loading")
            else:
                print(f"✅ Vector store verified: {total_courses} courses available")

        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️  Documents directory not found: {docs_path}")
        print("   This will cause 'query failed' errors until documents are loaded")

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    
# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")