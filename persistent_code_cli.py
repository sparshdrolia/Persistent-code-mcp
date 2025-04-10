#!/usr/bin/env python3
"""
Persistent-Code Command Line Interface

A command-line utility for managing the Persistent-Code MCP server.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import textwrap
import platform

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Persistent-Code CLI - Manage your Persistent-Code MCP server"
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
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the MCP server")
    start_parser.add_argument(
        "--project-name", "-p", 
        default="default", 
        help="Name of the project"
    )
    start_parser.add_argument(
        "--storage-dir", "-s", 
        default=None, 
        help="Directory to store persistent data"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show server status")
    
    # Configure command
    config_parser = subparsers.add_parser("config", help="Configure Claude for Desktop")
    config_parser.add_argument(
        "--project-name", "-p", 
        default="default", 
        help="Name of the project"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available projects")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze code files")
    analyze_parser.add_argument(
        "--project-name", "-p", 
        default="default", 
        help="Name of the project"
    )
    analyze_parser.add_argument(
        "--file", "-f", 
        required=True, 
        help="File to analyze"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    return args

def get_claude_config_path():
    """Get the path to the Claude for Desktop config file."""
    if platform.system() == "Darwin":  # macOS
        return os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
    elif platform.system() == "Windows":
        return os.path.join(os.environ.get("APPDATA", ""), "Claude", "claude_desktop_config.json")
    else:
        return os.path.expanduser("~/.config/Claude/claude_desktop_config.json")

def init_project(args):
    """Initialize a new project."""
    cmd = [sys.executable, "-m", "persistent_code", "init", "--project-name", args.project_name]
    
    if args.storage_dir:
        cmd.extend(["--storage-dir", args.storage_dir])
    
    print(f"Initializing project '{args.project_name}'...")
    subprocess.run(cmd)
    print(f"Project '{args.project_name}' initialized successfully.")
    print("\nTo start the server:")
    print(f"  {sys.executable} -m persistent_code serve --project-name {args.project_name}")
    
    # Suggest configuring Claude
    print("\nTo configure Claude for Desktop:")
    print(f"  {sys.executable} {__file__} config --project-name {args.project_name}")

def start_server(args):
    """Start the MCP server."""
    cmd = [sys.executable, "-m", "persistent_code", "serve", "--project-name", args.project_name]
    
    if args.storage_dir:
        cmd.extend(["--storage-dir", args.storage_dir])
    
    print(f"Starting server for project '{args.project_name}'...")
    print("Press Ctrl+C to stop the server.")
    print("")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")

def show_status():
    """Show the server status."""
    # Check if server is running (very basic)
    try:
        with open("/tmp/persistent_code_pid", "r") as f:
            pid = f.read().strip()
        
        # Check if process is running
        os.kill(int(pid), 0)
        print(f"Server is running (PID: {pid})")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        print("Server is not running")

def configure_claude(args):
    """Configure Claude for Desktop."""
    config_path = get_claude_config_path()
    config_dir = os.path.dirname(config_path)
    
    # Ensure config directory exists
    os.makedirs(config_dir, exist_ok=True)
    
    # Get absolute path to Python executable
    python_path = os.path.abspath(sys.executable)
    
    # Get absolute path to project directory
    project_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Load existing config if it exists
    config = {"mcpServers": {}}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            if "mcpServers" not in config:
                config["mcpServers"] = {}
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {config_path}. Creating a new config.")
    
    # Add Persistent-Code configuration
    config["mcpServers"]["persistent-code"] = {
        "command": python_path,
        "args": [
            "-m",
            "persistent_code",
            "serve",
            "--project-name",
            args.project_name
        ],
        "cwd": project_dir
    }
    
    # Write the config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Claude for Desktop configured to use project '{args.project_name}'")
    print(f"Configuration written to: {config_path}")
    print("\nPlease restart Claude for Desktop for the changes to take effect.")

def list_projects():
    """List available projects."""
    # Check storage directory for projects
    storage_dir = os.path.join(os.path.dirname(__file__), "storage")
    
    if not os.path.exists(storage_dir):
        print("No projects found. Storage directory doesn't exist.")
        return
    
    projects = []
    
    # Find all project directories with graph files
    for item in os.listdir(storage_dir):
        item_path = os.path.join(storage_dir, item)
        if os.path.isdir(item_path):
            # Check if project has a graph file
            graph_file = os.path.join(item_path, f"{item}_graph.json")
            config_file = os.path.join(item_path, "config.json")
            
            if os.path.exists(graph_file) or os.path.exists(config_file):
                projects.append(item)
    
    if not projects:
        print("No projects found.")
        return
    
    print("Available projects:")
    for project in sorted(projects):
        print(f"  - {project}")
    
    print("\nTo start a server for a project:")
    print(f"  {sys.executable} {__file__} start --project-name <project-name>")

def analyze_file(args):
    """Analyze a code file."""
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return
    
    # Read the file
    with open(args.file, "r") as f:
        code = f.read()
    
    # Get file path
    file_path = os.path.abspath(args.file)
    
    # Run the analyzer
    from persistent_code.knowledge_graph import KnowledgeGraph
    from persistent_code.code_analyzer import CodeAnalyzer
    
    # Create a knowledge graph
    graph = KnowledgeGraph(args.project_name)
    
    # Create a code analyzer
    analyzer = CodeAnalyzer(graph)
    
    # Analyze the code
    print(f"Analyzing {file_path}...")
    component_id = analyzer.analyze_code(
        code_text=code,
        file_path=file_path
    )
    
    # Print summary of what was found
    component = graph.get_component(component_id)
    print(f"\nFile analyzed: {component['name']}")
    print(f"Component ID: {component_id}")
    
    # Count nodes in the graph
    node_count = len(graph.graph.nodes)
    print(f"Total components in graph: {node_count}")
    
    # Print types of components found
    component_types = {}
    for _, data in graph.graph.nodes(data=True):
        comp_type = data.get("type", "unknown")
        component_types[comp_type] = component_types.get(comp_type, 0) + 1
    
    print("\nComponents by type:")
    for comp_type, count in component_types.items():
        print(f"  - {comp_type}: {count}")

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "init":
        init_project(args)
    elif args.command == "start":
        start_server(args)
    elif args.command == "status":
        show_status()
    elif args.command == "config":
        configure_claude(args)
    elif args.command == "list":
        list_projects()
    elif args.command == "analyze":
        analyze_file(args)

if __name__ == "__main__":
    main()
