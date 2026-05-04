"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract
89233218CNA000001 for Los Alamos National Laboratory (LANL),
which is operated by Triad National Security, LLC for the
U.S. Department of Energy/National Nuclear Security
Administration. All rights in the program are reserved by
Triad National Security, LLC, and the U.S. Department of
Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its
behalf a nonexclusive, paid-up, irrevocable worldwide
license in this material to reproduce, prepare. derivative
works, distribute copies to the public, perform publicly and
display publicly, and to permit others to do so.

cli.py
======
Command-line interface for the Tremors seismic agent.

Usage
-----
Run a one-shot query:

    tremors query "M5+ earthquakes in Japan in 2020"

LLM back-end selection
----------------------
The ``--backend`` flag chooses the LLM provider:

    ollama   (default) – local Ollama server via OpenAI-compatible endpoint
    openai             – OpenAI API (requires OPENAI_API_KEY env var)
    anthropic          – Anthropic API (requires ANTHROPIC_API_KEY env var) (TODO -RAD)

Use ``--model`` to override the default model for whichever backend is chosen.
Use ``--base-url`` to override the endpoint for the ollama backend (useful
when the server is on a remote machine).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional


logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("tremors")


# Default models per backend
_DEFAULT_MODELS = {
    "ollama":    "gpt-oss:20b",
    "openai":    "gpt-4o",
    # "anthropic": "TODO",
}

_DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


def _build_llm(
    backend:  str,
    model:    Optional[str],
    base_url: Optional[str],
    temperature: float,
):
    """
    Instantiate the appropriate LangChain chat model for *backend*.

    Raises SystemExit with a helpful message if a required environment
    variable or package is missing.
    """
    resolved_model = model or _DEFAULT_MODELS[backend]

    if backend == "ollama":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            sys.exit(
                "langchain-openai is required for the ollama backend.\n"
                "Install it with:  pip install langchain-openai"
            )
        return ChatOpenAI(
            base_url=base_url or _DEFAULT_OLLAMA_URL,
            api_key="ollama",
            model=resolved_model,
            temperature=temperature,
        )

    if backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Export it before running:  export OPENAI_API_KEY=sk-..."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            sys.exit(
                "langchain-openai is required for the openai backend.\n"
                "Install it with:  pip install langchain-openai"
            )
        return ChatOpenAI(
            api_key=api_key,
            model=resolved_model,
            temperature=temperature,
        )

    #TODO
    # if backend == "anthropic":
    #     api_key = os.environ.get("ANTHROPIC_API_KEY")
    #     if not api_key:
    #         sys.exit(
    #             "ANTHROPIC_API_KEY environment variable is not set.\n"
    #             "Export it before running:  export ANTHROPIC_API_KEY=sk-ant-..."
    #         )
    #     try:
    #         from langchain_anthropic import ChatAnthropic
    #     except ImportError:
    #         sys.exit(
    #             "langchain-anthropic is required for the anthropic backend.\n"
    #             "Install it with:  pip install langchain-anthropic"
    #         )
    #     return ChatAnthropic(
    #         api_key=api_key,
    #         model=resolved_model,
    #         temperature=temperature,
    #     )

    sys.exit(f"Unknown backend: {backend!r}. Choose from: ollama, openai") #, anthropic TODO


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent(args: argparse.Namespace):
    """Build the TremorsAgent from parsed CLI args."""
    try:
        from tremors import TremorsAgent
    except ImportError as exc:
        sys.exit(f"Could not import TremorsAgent: {exc}")

    llm = _build_llm(
        backend=args.backend,
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
        temperature=args.temperature,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    return TremorsAgent(llm=llm, output_dir=str(output_dir))

def _print_result(result: dict) -> None:
    """Print a human-readable summary of an agent result dict."""
    status = result.get("status", "Unknown")
    error  = result.get("error")

    print(f"\n{'─' * 60}")
    print(f"  Status : {status}")

    if error:
        print(f"  Error  : {error}")

    queried = result.get("queried_dcs") or []
    if queried:
        print(f"  DCs    : {', '.join(queried)}")

    tables = result.get("metadata_tables") or {}
    if tables:
        print(f"  Tables : {', '.join(tables.keys())}")
        for name, path in tables.items():
            print(f"           {name:20s} → {path}")

    plots = result.get("plots") or []
    for p in plots:
        print(f"  Plot   : {p}")

    waveforms = result.get("waveforms_saved") or result.get("continuous_waveforms_saved") or []
    if waveforms:
        print(f"  Waveforms saved : {len(waveforms)} file(s)")
        for w in waveforms[:5]:
            print(f"           {w}")
        if len(waveforms) > 5:
            print(f"           … and {len(waveforms) - 5} more")

    wplots = result.get("waveform_plots") or result.get("continuous_waveform_plots") or []
    for wp in wplots:
        print(f"  Waveform plot : {wp}")

    print(f"{'─' * 60}\n")



def _cmd_query(args: argparse.Namespace) -> None:
    """Run a single natural-language query and exit."""
    agent = _build_agent(args)

    inputs = {
        "query":      args.query,
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
    }

    print(f"[tremors] Running query: {args.query!r}")
    result = agent._action.invoke(inputs)
    _print_result(result)


def _build_parser() -> argparse.ArgumentParser:
    # ── Top-level parser ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="tremors",
        description=(
            "Tremors – FDSN seismic data retrieval and visualization agent.\n\n"
            "Examples:\n"
            "  tremors query \"M5+ earthquakes near Japan in 2020\"\n"
            "  tremors query \"continuous BH* waveforms for CI network, Feb 2016\" "
            "--output-dir ./ci_feb16 --backend ollama\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Shared options (apply to all sub-commands) ─────────────────────
    shared = argparse.ArgumentParser(add_help=False)

    shared.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        default="./tremors_output",
        metavar="DIR",
        help="Directory for output files (parquet, plots, MiniSEED). "
             "Created if it does not exist. Default: ./tremors_output",
    )
    shared.add_argument(
        "--backend", "-b",
        choices=["ollama", "openai"], #, anthropic TODO
        default="ollama",
        help="LLM backend to use. Default: ollama",
    )
    shared.add_argument(
        "--model", "-m",
        default=None,
        metavar="NAME",
        help=(
            "Model name to pass to the backend. "
            f"Defaults: ollama={_DEFAULT_MODELS['ollama']!r}, "
            f"openai={_DEFAULT_MODELS['openai']!r}, "
            # f"anthropic={_DEFAULT_MODELS['anthropic']!r}" TODO
        ),
    )
    shared.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        metavar="URL",
        help=f"Base URL for the ollama backend. Default: {_DEFAULT_OLLAMA_URL}",
    )
    shared.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        metavar="FLOAT",
        help="LLM sampling temperature. Default: 0.7",
    )
    shared.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging.",
    )

    # ── Sub-commands ───────────────────────────────────────────────────
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # tremors query <QUERY>
    p_query = sub.add_parser(
        "query",
        parents=[shared],
        help="Run a single natural-language query and exit.",
        description="Send one query to the Tremors agent and print the result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_query.add_argument(
        "query",
        metavar="QUERY",
        help='Natural-language seismic query, e.g. "M6+ earthquakes in Chile 2010-2020"',
    )
    p_query.set_defaults(func=_cmd_query)

    return parser




def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("tremors").setLevel(logging.DEBUG)

    args.func(args)


if __name__ == "__main__":
    main()