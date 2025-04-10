# Persistent-Code MCP Usage Guide

This guide explains how to use the Persistent-Code MCP server with Claude to maintain code context across sessions.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Starting a Project](#starting-a-project)
4. [Working with Claude](#working-with-claude)
5. [Real-World Examples](#real-world-examples)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.10 or higher
- Claude for Desktop (latest version)

### Using the Installation Script

The easiest way to install is using the provided script:

```bash
chmod +x install.sh
./install.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Initialize a default project
- Configure Claude for Desktop (if possible)

### Manual Installation

If you prefer to install manually:

```bash
# Clone the repository
git clone https://github.com/yourusername/persistent-code-mcp.git
cd persistent-code-mcp

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize a project
python -m persistent_code init --project-name "YourProject"
```

## Configuration

### Claude for Desktop Configuration

1. Edit your Claude for Desktop configuration file:
   - Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add the following configuration:

```json
{
  "mcpServers": {
    "persistent-code": {
      "command": "/path/to/venv/bin/python",
      "args": [
        "-m",
        "persistent_code",
        "serve",
        "--project-name",
        "YourProject"
      ],
      "cwd": "/path/to/persistent-code-mcp"
    }
  }
}
```

Replace `/path/to/venv/bin/python` with the actual path to your Python interpreter in the virtual environment, and `/path/to/persistent-code-mcp` with the actual path to the project directory.

3. Restart Claude for Desktop

## Starting a Project

### Creating a New Project

```bash
python -m persistent_code init --project-name "TodoApp"
```

This creates a new empty knowledge graph for your project.

### Starting the Server

```bash
python -m persistent_code serve --project-name "TodoApp"
```

The server will start and listen for requests from Claude.

## Working with Claude

### Initial Session

When you start working with Claude, begin by explaining the project:

**You:** "I want to build a Todo application with Python and FastAPI. Let's start by designing the data models and basic CRUD endpoints."

**Claude:** *Claude will design the components and store them in the knowledge graph.*

### Subsequent Sessions

In later sessions, you can ask Claude about the current state:

**You:** "What's the current status of our Todo app project?"

**Claude:** *Claude will query the knowledge graph and provide a summary of implemented and pending components.*

### Implementing Features

To implement specific features:

**You:** "Let's implement the feature to mark a todo item as completed."

**Claude:** *Claude will retrieve relevant context from the knowledge graph and provide consistent implementation.*

### Checking Project Status

To get an overview of implementation status:

**You:** "What's the current implementation status of the project?"

**Claude:** *Claude will use the `get_project_status` tool to provide statistics and summaries.*

### Navigating the Codebase

To explore components:

**You:** "Show me the TodoItem model and related components."

**Claude:** *Claude will retrieve components and their relationships from the knowledge graph.*

## Real-World Examples

### Example 1: Multi-Session Development

**Session 1:**

```
You: I want to build a FastAPI app for managing books. Let's start with designing the data models.

Claude: [Designs Book and Author models and stores them in the knowledge graph]
```

**Session 2:**

```
You: Let's continue with our book management app. What do we have so far?

Claude: [Retrieves models from knowledge graph]
We've designed the Book and Author models. Next, we should implement the API endpoints for CRUD operations.

You: Great, let's implement the endpoint for creating a new book.

Claude: [Uses context from the knowledge graph to provide consistent implementation]
```

### Example 2: Large Codebase Navigation

```
You: I'm working on a feature to add user ratings to books. Which components do I need to modify?

Claude: [Analyzes relationships in the knowledge graph]
You'll need to:
1. Update the Book model to include ratings
2. Create a Rating model for the many-to-many relationship
3. Add endpoints for adding/updating ratings
4. Update the book retrieval endpoint to include rating information
```

## Troubleshooting

### Server Not Starting

If the server fails to start:

1. Ensure Python 3.10+ is installed:
   ```bash
   python --version
   ```

2. Check that all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the storage directory exists and is writable:
   ```bash
   mkdir -p storage
   ```

### Claude Not Connecting to Server

If Claude doesn't connect to the MCP server:

1. Verify the server is running in a terminal window
2. Check the Claude for Desktop configuration file for errors
3. Restart Claude for Desktop
4. Look for error messages in the server console

### Incorrect or Missing Code Context

If Claude seems to have incorrect or missing information about the code:

1. Ensure the code was properly analyzed:
   ```
   You: Can you analyze this code for our project? [Paste code here]
   ```

2. Check component relationships:
   ```
   You: Show me the relationships between our components.
   ```

3. Rebuild context if necessary:
   ```
   You: Let's start fresh. Please analyze all of our code again to rebuild context.
   ```

## Advanced Features

### Custom Storage Directory

You can specify a custom storage directory:

```bash
python -m persistent_code init --project-name "CustomProject" --storage-dir "/path/to/storage"
python -m persistent_code serve --project-name "CustomProject" --storage-dir "/path/to/storage"
```

### Multiple Projects

You can work with multiple projects by specifying different project names:

```bash
python -m persistent_code init --project-name "ProjectA"
python -m persistent_code init --project-name "ProjectB"
```

And then serve the project you want to work on:

```bash
python -m persistent_code serve --project-name "ProjectA"
```
