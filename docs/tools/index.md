# Tools

DeepFabric uses [Spin](https://spinframework.dev), a WebAssembly framework, to execute tools during dataset generation. Tools run in isolated sandboxes, producing authentic training data based on real execution results.

## Why Real Execution Matters

Traditional synthetic data generators simulate tool outputs, which creates unrealistic training data. With Spin, tools execute against real state:

=== "Simulated (Unrealistic)"

    !!! danger "Hallucinated Results"
        The model invents tool outputs - file contents, API responses, command results - with no grounding in reality.

    ```python title="Traditional Approach"
    Agent: read_file("config.json")
    Result: {"setting": "value"}  # Made up

    Agent: api_call("GET /users/123")
    Result: {"name": "John", "id": 123}  # Fabricated

    Agent: run_command("git status")
    Result: "nothing to commit"  # Assumed
    ```

=== "Real Execution (Accurate)"

    !!! success "Grounded Results"
        Tools execute against real state - files, APIs, commands return actual results.

    ```python title="DeepFabric + Spin"
    Agent: read_file("config.json")
    Result: FileNotFound  # Actually doesn't exist

    Agent: write_file("config.json", '{"debug": true}')
    Result: Written 16 bytes  # Real operation

    Agent: read_file("config.json")
    Result: {"debug": true}  # Actual content
    ```

## Architecture

```mermaid
graph TB
    A[DeepFabric] --> B[Spin Service]
    B --> C[VFS<br/>Component]
    B --> D[Mock<br/>Component]
    B --> E[Custom<br/>Components]
```

Components are WebAssembly modules that handle specific tool categories:

| Component | Purpose | Tools |
|-----------|---------|-------|
| **VFS** | Virtual filesystem | read_file, write_file, list_files, delete_file |
| **Mock** | Dynamic mock execution | Any tool loaded via MCP |
| **Custom** | Your own components | Any tools you build (see [Custom Tools](custom.md)) |

## Quick Start

### Install Spin

=== "Docker"

    We provide a prepacked Docker image:

    ```bash
    docker run -d -p 3000:3000 ghcr.io/nolabs-ai/deepfabric/tools-sdk:latest
    ```

    This will now be accessible at `http://localhost:3000`.

=== "macOS"

    ```bash
    brew install fermyon/tap/spin
    ```

=== "Linux"

    ```bash
    curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash
    sudo mv spin /usr/local/bin/
    ```

=== "Local Build"

    From the `tools-sdk/` directory:

    ```bash
    spin build
    spin up
    ```

    The service runs at `http://localhost:3000`.

Configure DeepFabric to use it:

```yaml title="config.yaml"
generation:
  tools:
    spin_endpoint: "http://localhost:3000"
    components:
      builtin:  # VFS tools -> /vfs/execute
        - read_file
        - write_file
        - list_files
```

Each component routes to its own endpoint (`/{component}/execute`).

## Session Isolation

Each dataset sample gets an isolated session. Files created during one sample don't affect others:

```python
# Session A: Creates config.json
# Session B: config.json doesn't exist

# After sample generation, session is cleaned up
```

!!! info "Automatic Cleanup"
    DeepFabric automatically creates and cleans up sessions for each sample, ensuring isolation between training examples.

## Next Steps

<div class="grid cards" markdown>

-   :material-cog: **Spin Setup**

    ---

    Installation and running the tool service

    [:octicons-arrow-right-24: Setup guide](spin.md)

-   :material-folder-outline: **VFS Component**

    ---

    Virtual filesystem tools for file operations

    [:octicons-arrow-right-24: Learn more](vfs.md)

-   :material-test-tube: **MCP & Mock Tools**

    ---

    Import tools from MCP servers and configure mock responses

    [:octicons-arrow-right-24: Configure mocks](mock.md)

-   :material-wrench: **Custom Tools**

    ---

    Creating your own Spin components

    [:octicons-arrow-right-24: Build tools](custom.md)

</div>
