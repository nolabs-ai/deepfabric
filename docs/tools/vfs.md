# VFS Component

The Virtual Filesystem (VFS) component provides file operations in an isolated, session-scoped environment. Each session gets its own virtual filesystem that doesn't persist across samples.

## Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `read_file` | Read file content | `file_path` (string) |
| `write_file` | Write to file | `file_path` (string), `content` (string) |
| `list_files` | List all files | None |
| `delete_file` | Delete a file | `file_path` (string) |

!!! tip "Need another builtin tool?"
    If you need a generic tool that would see wide use, [open an issue](https://github.com/nolabs-ai/deepfabric/issues) to request it.

## Configuration

```yaml title="config.yaml"
generation:
  tools:
    spin_endpoint: "http://localhost:3000"
    components:
      builtin:  # Routes to /vfs/execute
        - read_file
        - write_file
        - list_files
```

!!! info "Builtin Component"
    The `builtin` component maps to VFS tools and routes to `/vfs/execute`. List specific tools or omit the list to include all builtin tools.

## Seeding Initial Files

Pre-populate files for scenarios:

```yaml title="config.yaml"
generation:
  tools:
    spin_endpoint: "http://localhost:3000"
    components:
      builtin:
        - read_file
        - write_file
        - list_files
    scenario_seed:
      files:
        "main.py": |
          def greet(name):
              return f"Hello, {name}!"

          if __name__ == "__main__":
              print(greet("World"))
        "config.json": |
          {
            "version": "1.0.0",
            "debug": true,
            "max_retries": 3
          }
```

The agent can then read and modify these files during generation.

## API Reference

### Execute Tool

```bash title="Execute request"
POST /vfs/execute
Content-Type: application/json

{
  "session_id": "sample-001",
  "tool": "read_file",
  "args": {"file_path": "config.json"}
}
```

### Response Format

```json title="Success response"
{
  "success": true,
  "result": "{\"debug\": true}",
  "error_type": null
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `FileNotFound` | File doesn't exist |
| `InvalidArguments` | Missing required parameter |
| `IOError` | Storage error |
