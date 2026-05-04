<div align="center">
  <img src="assets/tremors7.svg" alt="TREMORS Logo" width="500"/>
</div>

# TREMORS

TREMORS (Text Referenced Event Mapping and Output Renderer for Seismographs) is an agentic framework that leverages large language model reasoning within a constrained LangGraph execution graph to automate seismic data retrieval.

Natural language queries are translated into a structured intermediate schema, which drives a reproducible, auditable workflow for waveform (event-based/continuous) and metadata acquisition.

Approved for unlimited release LA-UR-26-23557

---

## Requirements

- Python ≥ 3.11
- [Ollama](https://ollama.com) (local models) or any OpenAI-compatible LLM backend (OpenAI, LiteLLM, etc.)

---

## Environment Setup

TREMORS supports both Conda and standard Python environments via `uv`.

### Option 1: Conda / Mamba (recommended)

```bash
conda env create --name tremors --file environment.yml
conda activate tremors
```

Or with Mamba:

```bash
mamba env create --name tremors --file environment.yml
mamba activate tremors
```

### Option 2: `uv` + Virtual Environment (lightweight)

Install `uv`:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Create and activate an environment:

```bash
uv venv --python 3.11
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

Install TREMORS:

```bash
uv pip install .
```

Optional dev dependencies:

```bash
uv pip install -e ".[dev]"
```

> **Notes:**
> - Dependencies are installed from `pyproject.toml`
> - System tools like Ollama must be installed separately

---

## Installation

After environment setup, install the package to make the `tremors` command available:

```bash
pip install -e .
tremors --help
```

---

## Ollama Setup

TREMORS supports local LLMs via Ollama. Choose the install method that fits your environment:

### Option 1: System Install (requires sudo)

```bash
curl -fsSL https://ollama.com/install.sh | bash
```

### Option 2: Local Install (no sudo)

```bash
bash scripts/install-ollama-nosudo.sh --install-dir /home/user/ollama
```

> Replace `/home/user/ollama` with your desired install path.

### Option 3: Conda

Ollama is included in `environment.yml` — if you used Conda/Mamba for setup, you already have it and can skip this step.

To install manually:

```bash
conda install -c conda-forge ollama
```

> **Note:** GPU acceleration may not be available depending on how the package was compiled. Use Options 1 or 2 for guaranteed GPU support.

### Start Ollama

Pull a model (only needs to be done once):

```bash
ollama pull gpt-oss:20b  # Or any model from https://ollama.com/library
```

Then start the server:

```bash
ollama serve &
```

> We've only tested with OpenAI-compatible models (e.g. `gpt-oss:20b`, `gpt-oss:120b`). Other models served by Ollama may work but are untested.

Check available models:

```bash
ollama list
```

---

## Usage

TREMORS is compatible with any OpenAI-style API backend:

| Backend | Endpoint | Env Var Required |
|---------|----------|-----------------|
| **Ollama** *(default)* | `http://localhost:11434/v1` | — |
| **LiteLLM** | Proxy for OpenAI, Anthropic, Gemini, etc. | Varies |
| **OpenAI** | Official API | `OPENAI_API_KEY` |

### Basic Example

> This example uses Ollama, but you can substitute any OpenAI-compatible provider — just swap `base_url`, `api_key`, and `model`.

```python
from langchain_openai import ChatOpenAI
from tremors import TremorsAgent

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",  # Replace with your provider's endpoint
    api_key="ollama",                      # Replace with your API key
    model="gpt-oss:20b",                   # Replace with your model name
    temperature=0.7,
)

agent = TremorsAgent(llm=llm, output_dir="./output")
result = agent.invoke({
    "query": "Retrieve waveforms for the 10 largest earthquakes along the Cascadia Subduction Zone between 2010 and 2020 and plot them."
})
```

---

## Command-Line Interface

TREMORS ships with a `tremors` CLI for one-shot natural-language queries against the agent.

### Installation

After installing the package the `tremors` command is available on your PATH:

```bash
pip install -e .
tremors --help
```

### Backends

| Flag | Provider | Env var required |
|------|----------|-----------------|
| `--backend ollama` *(default)* | Local Ollama server | — |
| `--backend openai` | OpenAI API | `OPENAI_API_KEY` |
<!-- | `--backend anthropic` | Anthropic API | `ANTHROPIC_API_KEY` | -->

---

### `tremors query` — One-shot Query

Send a single natural-language request and exit.

```
tremors query QUERY [options]
```

**Example — event catalog + waveforms + plots via NCEDC:**

```bash
tremors query "Find 2 unique events in Northern California from 2016 \
    with magnitude > 5.0. Get waveforms and plot them. \
    Use the NCEDC datacenter for everything." \
    --output-dir ./temp \
    --backend ollama \
    --model gpt-oss:20b \
    --base-url http://localhost:11438/v1
```
---

### All Options

```
usage: tremors query [-h] [-o DIR] [-b {ollama,openai,anthropic}]
                     [-m NAME] [--base-url URL] [-t FLOAT] [-v]
                     QUERY

options:
  -o, --output-dir DIR          Output directory. Default: ./tremors_output
  -b, --backend                 LLM backend. Default: ollama
  -m, --model NAME              Model name override
      --base-url URL            Ollama server URL. Default: http://localhost:11438/v1
  -t, --temperature FLOAT       Sampling temperature. Default: 0.7
  -v, --verbose                 Enable debug logging
```

---
## Useful Ollama Commands

```bash
ollama list         # list models
ollama rm <model>   # remove model
```

---

## Troubleshooting

### No GPU Detected

Ollama will fall back to CPU automatically. No action needed, but performance will be slower.

### Port Conflict (default port 11434 already in use)

Start Ollama on a different port:

```bash
OLLAMA_HOST=127.0.0.1:<PORT> ollama serve
```

> Replace `<PORT>` with an available port number (e.g. `11435`).

Then update your client config to match:

```python
base_url="http://localhost:<PORT>/v1"
```
