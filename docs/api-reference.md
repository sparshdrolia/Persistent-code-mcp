# Persistent-Code MCP API Reference

This document provides detailed information about all the tools exposed by the Persistent-Code MCP server.

## Tool Categories

1. [Knowledge Graph Management](#knowledge-graph-management)
2. [Code Retrieval and Navigation](#code-retrieval-and-navigation)
3. [Status Management](#status-management)
4. [Context Assembly](#context-assembly)
5. [Code Analysis](#code-analysis)

## Knowledge Graph Management

### add_component

Adds a new code component to the graph.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `code_text` | string | The actual code | Yes |
| `component_type` | string | Type of component (file, class, function, method, variable) | Yes |
| `name` | string | Name of the component | Yes |
| `status` | string | Implementation status (planned, partial, implemented) | No (default: "planned") |
| `description` | string | Semantic description of the component | No (default: "") |

**Returns:**

```json
{
  "success": true,
  "component_id": "uuid-string",
  "message": "Added {component_type} '{name}' with ID {component_id}"
}
```

### update_component

Updates an existing component in the graph.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `component_id` | string | ID of the component to update | Yes |
| `code_text` | string | New code text | No |
| `status` | string | New implementation status | No |
| `description` | string | New description | No |

**Returns:**

```json
{
  "success": true,
  "message": "Updated component {component_id}"
}
```

### add_relationship

Creates a relationship between components.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `source_id` | string | ID of the source component | Yes |
| `target_id` | string | ID of the target component | Yes |
| `relationship_type` | string | Type of relationship (imports, calls, inherits, contains, depends_on) | Yes |
| `metadata` | object | Additional metadata for the relationship | No |

**Returns:**

```json
{
  "success": true,
  "relationship_id": "uuid-string",
  "message": "Added {relationship_type} relationship from {source_id} to {target_id}"
}
```

## Code Retrieval and Navigation

### get_component

Retrieves a component by ID or name.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `identifier` | string | Component ID or name | Yes |
| `include_related` | boolean | Whether to include related components | No (default: false) |

**Returns:**

```json
{
  "success": true,
  "component": {
    "id": "uuid-string",
    "name": "ComponentName",
    "type": "function",
    "status": "implemented",
    "description": "Component description",
    "code_text": "function code...",
    "created_at": "ISO-datetime",
    "last_modified": "ISO-datetime",
    "version": 1,
    "metadata": {...}
  }
}
```

### find_related_components

Finds components related to a given component.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `component_id` | string | ID of the component | Yes |
| `relationship_types` | array | Types of relationships to consider | No |
| `depth` | number | How many levels of relationships to traverse | No (default: 1) |

**Returns:**

```json
{
  "success": true,
  "related_components": [
    {
      "id": "uuid-string",
      "name": "RelatedComponent",
      "type": "method",
      ...
    },
    ...
  ],
  "count": 2
}
```

### search_code

Searches the codebase semantically.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `query` | string | Search query | Yes |
| `component_types` | array | Types of components to search | No |
| `limit` | number | Maximum number of results | No (default: 10) |

**Returns:**

```json
{
  "success": true,
  "matches": [
    {
      "id": "uuid-string",
      "name": "MatchingComponent",
      "type": "function",
      ...
    },
    ...
  ],
  "count": 5
}
```

## Status Management

### update_status

Updates implementation status of a component.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `component_id` | string | ID of the component | Yes |
| `new_status` | string | New implementation status (planned, partial, implemented) | Yes |
| `notes` | string | Optional notes about the status change | No (default: "") |

**Returns:**

```json
{
  "success": true,
  "message": "Updated status of {component_id} to {new_status}"
}
```

### get_project_status

Retrieves implementation status across the project.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `filters` | object | Filters to apply to components | No |
| `grouping` | string | How to group the results (e.g., by type) | No |

**Returns:**

```json
{
  "success": true,
  "status": {
    "total_components": 25,
    "status_counts": {
      "planned": 10,
      "partial": 5,
      "implemented": 10
    },
    "implementation_percentage": 50.0,
    "components": [...]
  }
}
```

### find_next_tasks

Suggests logical next components to implement.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `priority_type` | string | How to prioritize tasks (dependencies, complexity) | No (default: "dependencies") |
| `limit` | number | Maximum number of tasks to suggest | No (default: 5) |

**Returns:**

```json
{
  "success": true,
  "tasks": [
    {
      "id": "uuid-string",
      "name": "NextComponent",
      "type": "function",
      ...
    },
    ...
  ],
  "count": 5
}
```

## Context Assembly

### prepare_context

Assembles minimal context for a specific task.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `task_description` | string | Description of the task | Yes |
| `relevant_components` | array | List of component IDs known to be relevant | No |
| `max_tokens` | number | Maximum tokens for the context | No (default: 4000) |

**Returns:**

```json
{
  "success": true,
  "context": {
    "task_description": "Implement feature X",
    "primary_components": [...],
    "related_components": [...],
    "summaries": {...}
  },
  "primary_count": 3,
  "related_count": 5,
  "summary_count": 2
}
```

### continue_implementation

Provides context to continue implementing a component.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `component_id` | string | ID of the component to continue implementing | Yes |
| `max_tokens` | number | Maximum tokens for the context | No (default: 4000) |

**Returns:**

```json
{
  "success": true,
  "context": {
    "component": {...},
    "dependencies": [...],
    "dependents": [...],
    "related_implementations": [...],
    "summaries": {...}
  }
}
```

### get_implementation_plan

Generates a plan for implementing pending components.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `component_ids` | array | List of component IDs to include | No (default: all pending) |
| `dependencies_depth` | number | How deep to analyze dependencies | No (default: 1) |

**Returns:**

```json
{
  "success": true,
  "plan": {
    "ordered_components": [...],
    "dependency_groups": {
      "group_1": [...],
      "group_2": [...]
    },
    "implementation_steps": [
      {
        "step": 1,
        "description": "Implement Group 1",
        "components": [...]
      },
      ...
    ]
  }
}
```

## Code Analysis

### analyze_code

Analyzes code and updates the knowledge graph.

**Parameters:**

| Name | Type | Description | Required |
|------|------|-------------|----------|
| `code_text` | string | The code to analyze | Yes |
| `file_path` | string | Path to the file | No |
| `component_id` | string | ID of an existing component to update | No |

**Returns:**

```json
{
  "success": true,
  "component_id": "uuid-string",
  "component": {...},
  "message": "Successfully analyzed and updated knowledge graph"
}
```
