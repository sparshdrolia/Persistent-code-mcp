"""
Main entry point for running the persistent-code MCP server.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

from .mcp_server import PersistentCodeMCP
from .config import config as config_instance

def parse_args() -> Dict[str, Any]:
    """Parse command line arguments.
    
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Persistent-Code MCP Server"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument(
        "--project-name", "-p", 
        default="default", 
        help="Name of the project"
    )
    init_parser.add_argument(
        "--storage-dir", "-s", 
        default=None, 
        help="Directory to store persistent data"
    )
    init_parser.add_argument(
        "--disable-llama-index", "-d",
        action="store_true",
        help="Disable LlamaIndex integration"
    )
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the MCP server")
    serve_parser.add_argument(
        "--project-name", "-p", 
        default="default", 
        help="Name of the project"
    )
    serve_parser.add_argument(
        "--storage-dir", "-s", 
        default=None, 
        help="Directory to store persistent data"
    )
    serve_parser.add_argument(
        "--transport", "-t", 
        default="stdio", 
        choices=["stdio", "http"],
        help="Transport protocol to use"
    )
    serve_parser.add_argument(
        "--port", 
        type=int,
        default=8000, 
        help="Port to use for HTTP transport"
    )
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configure settings")
    config_parser.add_argument(
        "--llama-index", "-l",
        choices=["enable", "disable"],
        help="Enable or disable LlamaIndex integration"
    )
    config_parser.add_argument(
        "--embedding-model", "-e",
        help="Set the embedding model for LlamaIndex"
    )
    config_parser.add_argument(
        "--similarity-top-k", "-k",
        type=int,
        help="Set the number of similar components to retrieve"
    )
    config_parser.add_argument(
        "--show-config", "-s",
        action="store_true",
        help="Show current configuration"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    return vars(args)

def init_project(project_name: str, storage_dir: str = None, disable_llama_index: bool = False) -> None:
    """Initialize a new project.
    
    Args:
        project_name: Name of the project
        storage_dir: Directory to store persistent data
        disable_llama_index: Whether to disable LlamaIndex integration
    """
    # Create storage directory if it doesn't exist
    storage_dir = storage_dir or os.path.join(os.getcwd(), "storage")
    os.makedirs(storage_dir, exist_ok=True)
    
    # Create project directory
    project_dir = os.path.join(storage_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    # Disable LlamaIndex if requested
    if disable_llama_index:
        config_instance.set("llama_index", "enabled", False)
        print("LlamaIndex integration disabled")
    
    print(f"Initialized project '{project_name}' in {project_dir}")

def configure_settings(args):
    """Configure settings.
    
    Args:
        args: Command-line arguments
    """
    # Show current configuration
    if args.get("show_config"):
        print("Current configuration:")
        print(f"  LlamaIndex enabled: {config_instance.is_llama_index_enabled()}")
        print(f"  Embedding model: {config_instance.get_embedding_model()}")
        print(f"  Similarity top-k: {config_instance.get_similarity_top_k()}")
        print(f"  Max tokens per component: {config_instance.get_max_tokens_per_component()}")
        print(f"  Logging level: {config_instance.get_logging_level()}")
        return
    
    # Update LlamaIndex setting
    if args.get("llama_index"):
        enabled = args["llama_index"] == "enable"
        config_instance.set("llama_index", "enabled", enabled)
        print(f"LlamaIndex integration {'enabled' if enabled else 'disabled'}")
    
    # Update embedding model
    if args.get("embedding_model"):
        config_instance.set("llama_index", "embedding_model", args["embedding_model"])
        print(f"Embedding model set to: {args['embedding_model']}")
    
    # Update similarity top-k
    if args.get("similarity_top_k"):
        config_instance.set("advanced", "similarity_top_k", args["similarity_top_k"])
        print(f"Similarity top-k set to: {args['similarity_top_k']}")

def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Initialize project
    if args["command"] == "init":
        init_project(args["project_name"], args["storage_dir"], args.get("disable_llama_index", False))
    
    # Start server
    elif args["command"] == "serve":
        # Create server
        server = PersistentCodeMCP(
            project_name=args["project_name"],
            storage_dir=args["storage_dir"]
        )
        
        # Print info
        print(f"Starting persistent-code MCP server for project '{args['project_name']}'")
        print(f"Transport: {args['transport']}")
        if args["transport"] == "http":
            print(f"Port: {args['port']}")
        
        # Run server
        if args["transport"] == "http":
            # TODO: Implement HTTP transport
            raise NotImplementedError("HTTP transport not yet implemented")
        else:
            server.run(transport="stdio")
    
    # Configure settings
    elif args["command"] == "config":
        configure_settings(args)

if __name__ == "__main__":
    main()
