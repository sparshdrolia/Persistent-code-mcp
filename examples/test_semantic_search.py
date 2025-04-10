#!/usr/bin/env python3
"""
Test Script for Semantic Search using LlamaIndex

This script demonstrates how to use the semantic search capabilities
of the Persistent-Code MCP server with LlamaIndex integration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persistent_code.knowledge_graph import (
    KnowledgeGraph,
    ComponentType,
    ComponentStatus
)
from persistent_code.code_analyzer import CodeAnalyzer
from persistent_code.config import config as config_instance
from persistent_code.llama_index_manager import LlamaIndexManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test semantic search capabilities"
    )
    
    # Project name
    parser.add_argument(
        "--project", "-p",
        default="test_project",
        help="Project name to use (default: test_project)"
    )
    
    # Storage directory
    parser.add_argument(
        "--storage-dir", "-d",
        help="Storage directory (default: ./storage)"
    )
    
    # Test file
    parser.add_argument(
        "--file", "-f",
        help="Test file to analyze"
    )
    
    # Search query
    parser.add_argument(
        "--query", "-q",
        help="Test search query"
    )
    
    # Enable/disable LlamaIndex
    parser.add_argument(
        "--enable-llama-index",
        action="store_true",
        help="Enable LlamaIndex integration"
    )
    
    parser.add_argument(
        "--disable-llama-index",
        action="store_true",
        help="Disable LlamaIndex integration"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Update LlamaIndex configuration if requested
    if args.enable_llama_index:
        config_instance.set("llama_index", "enabled", True)
        logger.info("LlamaIndex integration enabled")
    
    if args.disable_llama_index:
        config_instance.set("llama_index", "enabled", False)
        logger.info("LlamaIndex integration disabled")
    
    # Print current configuration
    llama_enabled = config_instance.is_llama_index_enabled()
    embedding_model = config_instance.get_embedding_model()
    
    print(f"LlamaIndex enabled: {llama_enabled}")
    print(f"Embedding model: {embedding_model}")
    
    # Create knowledge graph
    project_name = args.project
    storage_dir = args.storage_dir
    
    try:
        knowledge_graph = KnowledgeGraph(project_name, storage_dir=storage_dir)
        print(f"Created knowledge graph for project: {project_name}")
    except Exception as e:
        logger.error(f"Failed to create knowledge graph: {e}")
        return 1
    
    # Check LlamaIndex status
    if hasattr(knowledge_graph, 'llama_index_manager'):
        llama_status = knowledge_graph.llama_index_manager.get_status()
        print(f"LlamaIndex status: {llama_status}")
    else:
        print("LlamaIndex manager not available")
    
    # Analyze test file if provided
    if args.file:
        file_path = args.file
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return 1
        
        print(f"\nAnalyzing file: {file_path}")
        
        try:
            # Read the file
            with open(file_path, "r") as f:
                code_text = f.read()
            
            # Create analyzer
            analyzer = CodeAnalyzer(knowledge_graph)
            
            # Analyze code
            component_id = analyzer.analyze_code(
                code_text=code_text,
                file_path=file_path
            )
            
            print(f"Analysis complete. Component ID: {component_id}")
            
        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            return 1
    
    # Test search if query provided
    if args.query:
        query = args.query
        print(f"\nPerforming search for: '{query}'")
        
        # Try semantic search first
        if llama_enabled:
            print("Attempting semantic search...")
            
            try:
                # Direct test of semantic search if LlamaIndex manager available
                if hasattr(knowledge_graph, 'llama_index_manager') and knowledge_graph.llama_index_manager.is_available():
                    results = knowledge_graph.llama_index_manager.semantic_search(
                        query=query,
                        similarity_top_k=5
                    )
                    
                    if results:
                        print(f"Semantic search found {len(results)} results:")
                        for i, (score, result) in enumerate(results):
                            print(f"{i+1}. {result.get('metadata', {}).get('name', 'Unknown')} (Score: {score:.4f})")
                    else:
                        print("No semantic search results found.")
                else:
                    print("LlamaIndex manager not available or initialized.")
            except Exception as e:
                logger.error(f"Semantic search error: {e}")
        
        # Standard search through knowledge graph
        print("\nPerforming standard search...")
        results = knowledge_graph.search_code(
            query=query,
            limit=5
        )
        
        if results:
            print(f"Search found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"{i+1}. [{result.get('type')}] {result.get('name')}")
        else:
            print("No search results found.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
