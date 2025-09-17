# Visual Code Builder - Headless JSON API

This document describes the headless JSON API for the Visual Code Builder service, which provides visual programming capabilities for agent development.

## Base URL
```
http://localhost:8006
```

## Health Check

### GET /health
Returns the health status of the service.

**Response:**
```json
{
  "service": "visual_builder",
  "status": "healthy",
  "database": true,
  "cache": true,
  "active_projects": 0,
  "websocket_connections": 0,
  "timestamp": "2025-09-16T13:41:38.588Z"
}
```

## Project Management

### POST /projects
Create a new visual project.

**Request:**
```json
{
  "name": "My Project",
  "description": "Project description",
  "agent_id": "optional-agent-id",
  "template_id": "optional-template-id"
}
```

**Response:**
```json
{
  "id": "project-uuid",
  "name": "My Project",
  "description": "Project description",
  "components": {},
  "connections": [],
  "metadata": {
    "created_at": "2025-09-16T13:41:38.588Z",
    "status": "draft",
    "version": "1.0.0"
  }
}
```

### GET /projects
List all visual projects.

**Response:**
```json
{
  "projects": [
    {
      "id": "project-uuid",
      "name": "My Project",
      "description": "Project description",
      "components": {...},
      "connections": [...],
      "metadata": {...}
    }
  ],
  "count": 1,
  "active": ["project-uuid"]
}
```

### GET /projects/{project_id}
Get details of a specific project.

**Response:** Same as project creation response.

## Component Management

### GET /components/library
Get available components organized by category.

**Response:**
```json
{
  "categories": {
    "logic": {
      "condition": {"name": "Condition", "ports": {"in": 1, "out": 2}},
      "loop": {"name": "Loop", "ports": {"in": 1, "out": 1}},
      "function": {"name": "Function", "ports": {"in": 1, "out": 1}}
    },
    "data": {
      "input": {"name": "Input", "ports": {"in": 0, "out": 1}},
      "output": {"name": "Output", "ports": {"in": 1, "out": 0}},
      "transform": {"name": "Transform", "ports": {"in": 1, "out": 1}}
    },
    "ai": {
      "llm": {"name": "LLM", "ports": {"in": 1, "out": 1}},
      "classifier": {"name": "Classifier", "ports": {"in": 1, "out": 1}},
      "embedder": {"name": "Embedder", "ports": {"in": 1, "out": 1}}
    }
  },
  "total": 9
}
```

### POST /components/add
Add a component to a project.

**Request:**
```json
{
  "project_id": "project-uuid",
  "component_type": "logic",
  "component_id": "function",
  "position": {"x": 100, "y": 100},
  "config": {"optional": "config"}
}
```

**Response:**
```json
{
  "component_id": "function_xyz123",
  "status": "added"
}
```

## Connection Management

### POST /connections/create
Create a connection between components.

**Request:**
```json
{
  "project_id": "project-uuid",
  "source_id": "component-id-1",
  "source_port": "output",
  "target_id": "component-id-2",
  "target_port": "input"
}
```

**Response:**
```json
{
  "status": "connected"
}
```

## Code Generation

### POST /generate/code
Generate code from a visual project.

**Request:**
```json
{
  "project_id": "project-uuid",
  "language": "python",
  "optimize": true
}
```

**Response:**
```json
{
  "code_id": "code-uuid",
  "language": "python",
  "code": "#!/usr/bin/env python3\n# Generated code...",
  "lines": 45
}
```

### GET /download/{code_id}
Download generated code as a file.

**Response:** Plain text file with generated code.

## Templates

### GET /templates
Get available project templates.

**Response:**
```json
{
  "templates": [
    {
      "name": "Basic Agent",
      "components": [...],
      "connections": [...]
    }
  ],
  "categories": ["basic", "ml", "api", "workflow", "integration"]
}
```

## WebSocket Support

### WebSocket: /ws/{project_id}
Real-time collaboration endpoint for project updates.

**Messages:**
- Component moves: `{"type": "component_move", "component_id": "id", "position": {"x": 100, "y": 100}}`
- Selection changes: `{"type": "selection_change", "selected": ["id1", "id2"]}`
- AI assistance: `{"type": "ai_assist", "query": "help me..."}`

## Error Handling

All endpoints return appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (validation errors)
- 404: Not Found
- 500: Internal Server Error

Error responses include a `detail` field with error description.

## Example Usage

```python
import requests

# Create project
project = requests.post("http://localhost:8006/projects", json={
    "name": "My API",
    "description": "FastAPI application"
}).json()

project_id = project["id"]

# Add components
requests.post("http://localhost:8006/components/add", json={
    "project_id": project_id,
    "component_type": "data",
    "component_id": "input",
    "position": {"x": 50, "y": 50}
})

# Generate code
code_response = requests.post("http://localhost:8006/generate/code", json={
    "project_id": project_id,
    "language": "python"
}).json()

print(code_response["code"])
```

## Integration Notes

- The service integrates with shared database and cache systems
- Event-driven architecture with cross-service communication
- Supports real-time collaboration via WebSocket
- Generated code includes proper FastAPI scaffolding with Pydantic models
- Component validation ensures project integrity before code generation