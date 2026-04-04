# Contributing to OmniPulse

Thank you for your interest in contributing to OmniPulse! We welcome researchers, engineers, and data scientists to help improve our domain-agnostic time-series analysis tools.

## Architecture Overview
OmniPulse consists of two decoupled sub-systems:
1. **The Python Engine (`open-source-wst/`)**: This is the mathematical core, performing the Wavelet Scattering Transform, dimensionality reduction, and exposing MCP (Model Context Protocol) endpoints.
2. **The TypeScript Orchestrator (`agent/`)**: The intelligent agentic loop that dynamically evaluates data thresholds, processes prompts, and interacts with the Python backend via MCP `stdio` transport.

## How to Contribute

### 1. Adding New MCP Tools
To expose a new analytical pipeline from Python to the TypeScript Agent:
1. Navigate to `open-source-wst/src/transient_wst/mcp_server.py`.
2. Define a new function decorated with `@mcp.tool()`. Ensure type hints and a comprehensive docstring are provided.
3. Keep I/O structured around `.npy` files and JSON logs.
4. On the TypeScript side (`agent/src/main.ts` / `QueryEngine`), the agent will automatically discover the new tool during the MCP handshake. Just ensure the agent prompt gives context on when to use it!

### 2. Expanding the Wavelet Backend
To implement a novel wavelet transform (e.g., CWT or DWT replacements) or new PyTorch accelerated modules:
1. Define the logic in `open-source-wst/src/transient_wst/core.py` or create a new module.
2. Adhere to strict typing standards (`mypy --strict`).
3. Include tests in `open-source-wst/tests/`. We enforce >90% coverage for the mathematical core.

### 3. Setting Up Your Development Environment

#### Python Engine
```bash
cd open-source-wst
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/
```

#### TypeScript Orchestrator
```bash
cd agent
bun install
bun test
```

## Pull Request Guidelines
- Always format Python code with `ruff` and verify types with `mypy`.
- Provide tests for any new signal processing logic.
- PR titles should follow conventional commits: `feat:`, `fix:`, `docs:`, etc.
- By submitting a PR, you agree your code will be released under the Apache 2.0 License.
