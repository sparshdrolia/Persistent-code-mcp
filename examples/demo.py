"""
Demo script for the Persistent-Code MCP server.

This script demonstrates how to use the Persistent-Code MCP server
to analyze Python code and maintain a knowledge graph.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import persistent_code
sys.path.insert(0, str(Path(__file__).parent.parent))

from persistent_code.knowledge_graph import (
    KnowledgeGraph, 
    ComponentType, 
    ComponentStatus,
    RelationshipType
)
from persistent_code.code_analyzer import CodeAnalyzer
from persistent_code.context_assembler import ContextAssembler

# Sample code for demo
TODO_APP_CODE = """
from fastapi import FastAPI, HTTPException, Depends, status
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from typing import List, Optional
import datetime

# Create FastAPI app
app = FastAPI(title="TodoApp API")

# SQLAlchemy setup
Base = declarative_base()

class TodoItem(Base):
    \"\"\"Model for todo items in the database.\"\"\"
    __tablename__ = "todo_items"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
# API routes
@app.get("/todos/", response_model=List[TodoItem])
def get_todos(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    \"\"\"Get all todo items.\"\"\"
    return db.query(TodoItem).offset(skip).limit(limit).all()

@app.post("/todos/", response_model=TodoItem, status_code=status.HTTP_201_CREATED)
def create_todo(todo: TodoItem, db: Session = Depends(get_db)):
    \"\"\"Create a new todo item.\"\"\"
    db_todo = TodoItem(**todo.dict())
    db.add(db_todo)
    db.commit()
    db.refresh(db_todo)
    return db_todo

@app.get("/todos/{todo_id}", response_model=TodoItem)
def get_todo(todo_id: int, db: Session = Depends(get_db)):
    \"\"\"Get a specific todo item by ID.\"\"\"
    db_todo = db.query(TodoItem).filter(TodoItem.id == todo_id).first()
    if db_todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")
    return db_todo

@app.put("/todos/{todo_id}", response_model=TodoItem)
def update_todo(todo_id: int, todo: TodoItem, db: Session = Depends(get_db)):
    \"\"\"Update a todo item.\"\"\"
    db_todo = db.query(TodoItem).filter(TodoItem.id == todo_id).first()
    if db_todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")
    
    for key, value in todo.dict().items():
        setattr(db_todo, key, value)
    
    db.commit()
    db.refresh(db_todo)
    return db_todo

@app.delete("/todos/{todo_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_todo(todo_id: int, db: Session = Depends(get_db)):
    \"\"\"Delete a todo item.\"\"\"
    db_todo = db.query(TodoItem).filter(TodoItem.id == todo_id).first()
    if db_todo is None:
        raise HTTPException(status_code=404, detail="Todo not found")
    
    db.delete(db_todo)
    db.commit()
    return {"ok": True}

def get_db():
    \"\"\"Get database session.\"\"\"
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
"""

def main():
    """Run the demo."""
    # Create a project directory
    project_dir = Path("demo_project")
    project_dir.mkdir(exist_ok=True)
    
    # Create a knowledge graph
    print("Creating knowledge graph...")
    graph = KnowledgeGraph("demo", storage_dir=str(project_dir))
    
    # Create a code analyzer
    analyzer = CodeAnalyzer(graph)
    
    # Analyze the TODO app code
    print("Analyzing code...")
    todo_file_id = analyzer.analyze_code(
        code_text=TODO_APP_CODE,
        file_path="todo_app.py"
    )
    
    # Print information about components
    print("\nComponents in the knowledge graph:")
    components = {}
    for node_id, data in graph.graph.nodes(data=True):
        name = data.get("name", "Unknown")
        comp_type = data.get("type", "Unknown")
        components[node_id] = (name, comp_type)
        print(f"- [{comp_type}] {name} (ID: {node_id})")
    
    # Create context assembler
    context_assembler = ContextAssembler(graph)
    
    # Prepare context for implementing a new feature
    print("\nPreparing context for implementing 'mark_completed' feature...")
    context = context_assembler.prepare_context(
        task_description="Implement an endpoint to mark a todo item as completed",
        max_tokens=2000
    )
    
    # Print the primary components in the context
    print("\nPrimary components for the task:")
    for component in context["primary_components"]:
        name = component.get("name", "Unknown")
        comp_type = component.get("type", "Unknown")
        print(f"- [{comp_type}] {name}")
    
    # Generate an implementation plan
    print("\nGenerating implementation plan...")
    plan = context_assembler.get_implementation_plan()
    
    # Print implementation steps
    print("\nImplementation steps:")
    for step in plan["implementation_steps"]:
        print(f"Step {step['step']}: {step['description']}")
        for component in step["components"]:
            print(f"  - {component['name']} ({component['type']})")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
