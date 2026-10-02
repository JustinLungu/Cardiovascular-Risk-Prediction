# Cardiovascular-Risk-Prediction
From-scratch cardiovascular risk prediction on 300k+ BRFSS records using NumPy, with custom ML algorithms, preprocessing, and cross-validation.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or via a package manager:

```bash
# macOS (Homebrew)
brew install uv

# pipx
pipx install uv
```

Verify the install:

```bash
uv --version
```

### 2. Set up the project environment

From the repo root:

```bash
uv sync
```

This creates a `.venv/` using Python 3.9, NumPy 1.23.1, and Matplotlib 3.5.2, matching the course grading environment. `uv.lock` pins the resolved dependencies. Development tools (pytest 7.1.2, GitPython 3.1.18, and Black 22.6.0) are included by default for the supplied tests; they are not libraries for the ML implementation. Use `uv sync --no-dev` for just the project dependencies.

### 3. Run code

```bash
uv run python your_script.py
```

or activate the environment directly:

```bash
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

## Download the dataset

To download and prepare the dataset, run:

```bash
bash scripts/prepare_data.sh
```
