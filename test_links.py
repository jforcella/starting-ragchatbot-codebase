#!/usr/bin/env python3
"""Test script to verify that lesson links are being stored and retrieved correctly"""

import sys
import os
sys.path.append('./backend')

from config import config
from vector_store import VectorStore

def test_lesson_links():
    """Test that lesson links are being retrieved properly"""
    print("Testing lesson link functionality...")
    
    # Initialize vector store
    vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL)
    
    # Get all courses metadata
    courses_metadata = vector_store.get_all_courses_metadata()
    print(f"\nFound {len(courses_metadata)} courses:")
    
    for course in courses_metadata:
        print(f"\nCourse: {course.get('title', 'Unknown')}")
        print(f"Course Link: {course.get('course_link', 'None')}")
        
        lessons = course.get('lessons', [])
        print(f"Lessons: {len(lessons)}")
        
        for lesson in lessons[:3]:  # Show first 3 lessons
            lesson_num = lesson.get('lesson_number')
            lesson_title = lesson.get('lesson_title', 'Unknown')
            lesson_link = lesson.get('lesson_link', 'None')
            print(f"  Lesson {lesson_num}: {lesson_title}")
            print(f"    Link: {lesson_link}")
            
            # Test the get_lesson_link method
            retrieved_link = vector_store.get_lesson_link(course['title'], lesson_num)
            print(f"    Retrieved: {retrieved_link}")
            print(f"    Match: {retrieved_link == lesson_link}")
        
        if len(lessons) > 3:
            print(f"  ... and {len(lessons) - 3} more lessons")

if __name__ == "__main__":
    test_lesson_links()