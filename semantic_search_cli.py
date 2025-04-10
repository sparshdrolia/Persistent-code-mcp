#!/usr/bin/env python3
"""
Semantic Search CLI for Persistent-Code MCP Server

A command-line interface for performing semantic code searches using 
the Persistent-Code MCP server with LlamaIndex integration.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persistent_code.knowledge_graph import (
    KnowledgeGraph,
    ComponentType,
    ComponentStatus
)
from persistent_code.code_analyzer import CodeAnalyzer
from persistent_code.config import config as config_instance

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Semantic Search CLI for Persistent-Code MCP"
    )
    
    # Project name
    parser.add_argument(
        "--project", "-p",
        required=True,
        help="Project name to search in"
    )
    
    # Semantic search vs. basic search
    parser.add_argument(
        "--semantic", "-s",
        action="store_true",
        help="Use semantic search (default: False, use basic text search)"
    )
    
    # Query
    parser.add_argument(
        "query",
        nargs="*",
        help="Search query (use quotes for multi-word queries)"
    )
    
    # Output format (text or json)
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    # Max results
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=5,
        help="Maximum number of results (default: 5)"
    )
    
    # Component types
    parser.add_argument(
        "--types", "-t",
        nargs="*",
        choices=["file", "class", "function", "method", "variable", "module"],
        help="Filter by component types"
    )
    
    # Interactive mode
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode (prompt for queries)"
    )
    
    # Analyze a file
    parser.add_argument(
        "--analyze", "-a",
        help="Analyze a file and add it to the knowledge graph"
    )
    
    # Storage directory
    parser.add_argument(
        "--storage-dir", "-d",
        help="Storage directory (default: ./storage)"
    )
    
    return parser.parse_args()

def format_component(component: Dict[str, Any], verbose: bool = False) -> str:
    """Format a component for display.
    
    Args:
        component: Component data
        verbose: Whether to show full details
        
    Returns:
        Formatted string
    """
    name = component.get("name", "Unnamed")
    component_type = component.get("type", "unknown")
    description = component.get("description", "")
    status = component.get("status", "unknown")
    
    if verbose:
        # Full details
        code_text = component.get("code_text", "")
        created_at = component.get("created_at", "")
        last_modified = component.get("last_modified", "")
        
        return (
            f"Name: {name}\n"
            f"Type: {component_type}\n"
            f"Status: {status}\n"
            f"Description: {description}\n"
            f"Created: {created_at}\n"
            f"Modified: {last_modified}\n"
            f"Code:\n{code_text}\n"
        )
    else:
        # Brief details
        return f"[{component_type}] {name}: {description[:100]}{'...' if len(description) > 100 else ''}"

def search(graph: KnowledgeGraph, query: str, limit: int = 5, component_types: Optional[List[str]] = None):
    """Search the knowledge graph.
    
    Args:
        graph: Knowledge graph
        query: Search query
        limit: Maximum number of results
        component_types: Types of components to search
    
    Returns:
        List of matching components
    """
    # Convert string component types to enum
    enum_types = None
    if component_types:
        enum_types = [ComponentType(t) for t in component_types]
    
    # Perform search
    return graph.search_code(
        query=query,
        component_types=enum_types,
        limit=limit
    )

def main():
    """Main entry point."""
    args = parse_args()
    
    # Get project name
    project_name = args.project
    
    # Get storage directory
    storage_dir = args.storage_dir
    
    # Check if LlamaIndex is enabled
    semantic_enabled = config_instance.is_llama_index_enabled()
    if args.semantic and not semantic_enabled:
        logger.warning("LlamaIndex integration is disabled in configuration. Using basic search.")
    
    # Create knowledge graph
    try:
        graph = KnowledgeGraph(project_name, storage_dir=storage_dir)
        logger.info(f"Loaded knowledge graph for project: {project_name}")
    except Exception as e:
        logger.error(f"Failed to load knowledge graph: {e}")
        return 1
    
    # Analyze a file if requested
    if args.analyze:
        try:
            file_path = args.analyze
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return 1
            
            with open(file_path, "r") as f:
                code_text = f.read()
            
            analyzer = CodeAnalyzer(graph)
            component_id = analyzer.analyze_code(
                code_text=code_text,
                file_path=file_path
            )
            
            logger.info(f"Analyzed file: {file_path}")
            logger.info(f"Component ID: {component_id}")
            
        except Exception as e:
            logger.error(f"Failed to analyze file: {e}")
            return 1
    
    # Interactive mode
    if args.interactive:
        print(f"Semantic Search CLI for Persistent-Code MCP - Project: {project_name}")
        print(f"Using {'semantic' if semantic_enabled and args.semantic else 'basic'} search")
        print("Enter a query (or 'exit' to quit):")
        
        while True:
            try:
                query = input("> ")
                if query.lower() in ("exit", "quit", "q"):
                    break
                
                # Skip empty queries
                if not query.strip():
                    continue
                
                # Perform search
                results = search(
                    graph=graph,
                    query=query,
                    limit=args.limit,
                    component_types=args.types
                )
                
                # Display results
                if not results:
                    print("No matching components found.")
                else:
                    print(f"\nFound {len(results)} matching components:")
                    for i, result in enumerate(results):
                        print(f"{i+1}. {format_component(result)}")
                        
                    # Ask if user wants to see details of a result
                    while True:
                        detail_input = input("\nShow details for a result? (number/n) > ")
                        if detail_input.lower() in ("n", "no", ""):
                            break
                        
                        try:
                            index = int(detail_input) - 1
                            if 0 <= index < len(results):
                                print("\n" + format_component(results[index], verbose=True))
                            else:
                                print(f"Invalid result number. Please enter 1-{len(results)}")
                        except ValueError:
                            print("Invalid input. Please enter a number or 'n'")
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
        
        print("Goodbye!")
        return 0
    
    # Regular search mode
    if args.query:
        query = " ".join(args.query)
        
        # Perform search
        results = search(
            graph=graph,
            query=query,
            limit=args.limit,
            component_types=args.types
        )
        
        # Output results
        if args.format == "json":
            # JSON output
            json_results = {
                "query": query,
                "count": len(results),
                "results": results
            }
            print(json.dumps(json_results, indent=2))
        else:
            # Text output
            if not results:
                print("No matching components found.")
            else:
                print(f"Found {len(results)} matching components:")
                for i, result in enumerate(results):
                    print(f"{i+1}. {format_component(result)}")
    else:
        # No query provided
        print("No query provided. Use --interactive for interactive mode.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
