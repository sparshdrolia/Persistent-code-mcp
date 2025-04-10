#!/bin/bash
# Installation script for Persistent-Code MCP Server

# Exit on error
set -e

echo "Persistent-Code MCP Server - Installation"
echo "----------------------------------------"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]); then
    echo "Error: Python 3.10 or higher is required (found $python_version)"
    echo "Please install a compatible Python version and try again."
    exit 1
fi

echo "Python version $python_version detected. OK!"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "persistent_code" ]; then
    echo "Error: Please run this script from the persistent-code-mcp directory."
    exit 1
fi

# Create directories
mkdir -p storage

# Set up virtual environment
echo "Setting up virtual environment..."

# Check if UV is installed
if command -v uv &> /dev/null; then
    echo "Using UV package manager"
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
else
    echo "UV not found, using pip"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Initialize a default project
echo "Initializing default project..."
python -m persistent_code init --project-name "default"

# Create config directory for Claude Desktop if it doesn't exist
CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
if [ ! -d "$CLAUDE_CONFIG_DIR" ]; then
    echo "Creating Claude Desktop config directory..."
    mkdir -p "$CLAUDE_CONFIG_DIR"
fi

# Determine absolute path to project
PROJECT_PATH=$(pwd)

# Check if Claude Desktop config exists
CONFIG_FILE="$CLAUDE_CONFIG_DIR/claude_desktop_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating Claude Desktop config file..."
    cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "persistent-code": {
      "command": "${PROJECT_PATH}/venv/bin/python",
      "args": [
        "-m",
        "persistent_code",
        "serve",
        "--project-name",
        "default"
      ],
      "cwd": "${PROJECT_PATH}"
    }
  }
}
EOF
else
    echo "Claude Desktop config file exists."
    echo "Please add the following to $CONFIG_FILE manually:"
    echo ""
    echo '  "persistent-code": {'
    echo '    "command": "'"${PROJECT_PATH}/venv/bin/python"'",'
    echo '    "args": ['
    echo '      "-m",'
    echo '      "persistent_code",'
    echo '      "serve",'
    echo '      "--project-name",'
    echo '      "default"'
    echo '    ],'
    echo '    "cwd": "'"${PROJECT_PATH}"'"'
    echo '  }'
fi

echo ""
echo "Installation complete!"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate  # or source .venv/bin/activate if using UV"
echo "  python -m persistent_code serve"
echo ""
echo "If you've configured Claude Desktop, restart it to connect to the Persistent-Code MCP server."
echo "You can then ask Claude to help you with coding projects with persistent context."
