# Quick start

!!! note "Fork Notice"

    This is a fork of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) with jeeves-core integration.
    For the original project, see [mini-swe-agent.com](https://mini-swe-agent.com).

## Installation

### From Source (Recommended for this fork)

Clone with submodule to get jeeves-core:

```bash
git clone --recursive https://github.com/Jeeves-Cluster-Organization/jeeves-capability-mini-swe-agent.git
cd jeeves-capability-mini-swe-agent
pip install -e ".[dev]"
```

Then run:

```bash
mini  # simple UI (legacy mode)
mini -v  # visual UI (legacy mode)
mini-jeeves -t "Fix the bug" --mode parallel  # jeeves integration mode
```

### Original mini-swe-agent (upstream)

If you want the original mini-swe-agent without jeeves-core:

```bash
pip install mini-swe-agent
```

See the [original documentation](https://mini-swe-agent.com/latest/quickstart/) for upstream installation options.

## Development Setup

If you are planning to contribute, install dev dependencies and `pre-commit` hooks:

```bash
pip install -e '.[dev]'
pip install pre-commit && pre-commit install
```

To check your installation, run `pytest -n auto` in the root folder.

!!! example "Example Prompts"

    Try mini-SWE-agent with these example prompts:

    - Implement a Sudoku solver in python in the `sudoku` folder. Make sure the codebase is modular and well tested with pytest.
    - Please run pytest on the current project, discover failing unittests and help me fix them. Always make sure to test the final solution.
    - Help me document & type my codebase by adding short docstrings and type hints.

## Models

!!! note "Models should be set up the first time you run `mini`"

    * If you missed the setup wizard, just run `mini-extra config setup`
    * For more information, please check the [model setup quickstart](models/quickstart.md).
    * If you want to use local models, please check this [guide](models/local_models.md).

    Tip: Please always include the provider in the model name, e.g., `anthropic/claude-...`.

!!! success "Which model to use?"

    We recommend using `anthropic/claude-sonnet-4-5-20250929` for most tasks.
    For openai models, we recommend using `openai/gpt-5` or `openai/gpt-5-mini`.
    You can check scores of different models at our [SWE-bench (bash-only)](https://swebench.com) leaderboard.
