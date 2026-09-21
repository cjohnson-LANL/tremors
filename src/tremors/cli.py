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
    anthropic          – Anthropic (or Anthropic-compatible gateway). Requires
                         ANTHROPIC_API_KEY plus a base URL and model, each given
                         by flag or env var (--base-url / ANTHROPIC_BASE_URL,
                         --model / ANTHROPIC_MODEL) — there is no default model.
                         Install the extra:  pip install -e ".[anthropic]"

Use ``--model`` to override the default model (ollama/openai) or supply it
(anthropic). Use ``--base-url`` to set the endpoint: it overrides the ollama
default and is required for anthropic.

Before the query runs, a preflight check pings the backend and exits with a
clear message if the credentials/endpoint/model don't work; pass
``--no-preflight`` to skip it.

Human-in-the-loop
-----------------
The agent pauses for approval before it retrieves anything: once before the event
catalog query, and once before a bulk continuous download. Each pause prints the
resolved search parameters and offers approve / edit / reject / quit. A plan that
fails verification instead asks for a clarification and re-plans.

``--yes`` approves every gate automatically (a clarification request still stops
the run — it cannot be answered on the user's behalf). ``--no-interrupts``
disables the gates altogether; verification still runs.

Driving Tremors from another program or agent harness
-----------------------------------------------------
The CLI is designed to be driven headlessly. Four properties make that work:

**1. Machine-readable output.** ``--json`` writes exactly one JSON document to
**stdout** and nothing else. Every human-facing line — progress logs, summaries,
gate prompts — goes to **stderr**, so stdout stays parseable even with ``-v``.
Artifact lists in the JSON are complete (the human summary truncates them).

**2. Meaningful exit codes.**

===== ============================ ==========================================
Code  Meaning                      Harness reaction
===== ============================ ==========================================
0     success / no_data            Read the artifact paths.
1     the run failed               Read ``error``; retry or report.
2     usage error (bad flags)      Fix the invocation.
3     clarification required       Rewrite the query with the missing detail.
4     awaiting input at a gate     Inspect ``interrupt``; ``tremors resume``.
5     config/backend error         Fix credentials, endpoint, model, deps.
6     aborted at an interactive    Nothing to do; the user declined.
      prompt
===== ============================ ==========================================

**3. Plan-only dry runs.** ``--plan-only`` resolves the query to verified
``search_params`` and reports the datacenters that *would* be queried, without
sending a single FDSN request or writing a file. Costs one planner LLM call.
Verifier findings come back in ``problems`` instead of pausing.

**4. Durable sessions for per-gate control.** ``--session-id NAME`` runs against
a SQLite checkpointer under ``~/.tremors`` (override with ``--session-dir`` or
``TREMORS_SESSION_DIR``), so a gate pause survives process exit. The run stops
with exit 4 and the pending request under ``interrupt``; a *later* invocation
answers it::

    tremors query "M5+ near Japan in 2020" --json --session-id s1
    #  → exit 4, {"outcome": "awaiting_input", "interrupt": {...}}

    tremors resume --session-id s1 --json --show          # re-read the request
    tremors resume --session-id s1 --json --approve
    tremors resume --session-id s1 --json --reject "too broad"
    tremors resume --session-id s1 --json --clarify "within 100 km of Tokyo"
    tremors resume --session-id s1 --json \
        --decision '{"type":"edit","edited_action":{"name":"query_cascade",
                     "args":{"params_override":{"min_mag":6.0}}}}'

``resume`` needs no backend flags: the session records the backend, model, base
URL, temperature and output directory of the run that created it (never the API
key — that always comes from the environment).

Without ``--session-id``, a non-interactive invocation that hits a gate still
reports it (exit 4) but cannot be resumed — the in-memory checkpointer dies with
the process — so pass ``-y`` to approve automatically or ``--session-id`` to
retain per-gate control.

**5. Stored settings, so an invocation needs no credential plumbing.** The
default backend is ``anthropic``, which has no built-in endpoint or model (it
targets arbitrary Anthropic-compatible gateways). Every setting resolves in one
order — **CLI flag > environment variable > config file > built-in default** —
and ``tremors config`` both reports the effective value with its source and
stores the values that would otherwise be exported every session::

    tremors config --save --base-url https://gateway.example \\
                   --model my-model-id --api-key sk-...
    tremors config --save --api-key -     # read the key from stdin instead
    tremors config --json                 # effective settings + "ready": bool

The file is ``~/.tremors/config.json`` (``--config`` / ``TREMORS_CONFIG``),
written mode ``0600``. It sits outside the repository, so a real key or a private
endpoint can be stored without any chance of committing it, and the stored key is
never echoed back — only whether one is set. When a required setting is missing
the run exits 5 with ``missing`` naming each one, plus how to store it and how to
switch backends instead.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional


logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger("tremors")


# Default models per backend. The anthropic backend deliberately has NO default:
# it targets arbitrary Anthropic-compatible gateways whose model ids vary (e.g.
# the LANL AI Portal needs "aws-gov.claude-opus-4-8", not the public
# "claude-opus-4-8"), so the model must be given explicitly (--model or
# ANTHROPIC_MODEL). See _build_llm.
_DEFAULT_MODELS = {
    "ollama":    "gpt-oss:20b",
    "openai":    "gpt-4o",
}

_DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
_DEFAULT_OUTPUT_DIR = "./tremors_output"
_DEFAULT_BACKEND    = "anthropic"

# The supported backends, in one place: argparse's --backend choices and the
# config file's per-backend sections validate against the same tuple, so a name
# accepted on the command line can always be stored and vice versa.
_BACKENDS = ("anthropic", "ollama", "openai")


# ---------------------------------------------------------------------------
# Machine-readable contract: schema version, outcomes, exit codes
# ---------------------------------------------------------------------------

# Bump when the shape of the --json document changes incompatibly. A harness can
# refuse to parse a document whose schema_version it does not recognise.
_SCHEMA_VERSION = 1

# Exit codes. 2 is reserved for argparse's own usage errors, so it is absent
# here; every other code is produced by _finish() or _die().
_EXIT_OK       = 0
_EXIT_FAILED   = 1
_EXIT_USAGE    = 2
_EXIT_CLARIFY  = 3
_EXIT_AWAITING = 4
_EXIT_CONFIG   = 5
_EXIT_ABORTED  = 6

_EXIT_FOR_OUTCOME = {
    "success":                _EXIT_OK,
    "no_data":                _EXIT_OK,
    "failed":                 _EXIT_FAILED,
    "unknown":                _EXIT_FAILED,
    "clarification_required": _EXIT_CLARIFY,
    "awaiting_input":         _EXIT_AWAITING,
    "config_error":           _EXIT_CONFIG,
    "aborted":                _EXIT_ABORTED,
}

# The agent's ``status`` strings are authored by whichever tool ran last, so a
# successful catalog+plot run ends as "Plots Generated" rather than "Success".
# Normalise that vocabulary to a small closed set a harness can branch on, and
# keep the raw string alongside it in the document.
_OUTCOME_BY_STATUS = {
    "success":                             "success",
    "success (no data)":                   "no_data",
    # The plotting node reports this whenever nothing was plottable — an empty
    # cascade result as much as events lacking magnitudes. Either way the run
    # produced no artifacts to read, which is what no_data tells a caller. Both
    # outcomes exit 0; only the label differs.
    "success (no events with magnitude)":  "no_data",
    "plots generated":                     "success",
    "query parsed":                        "success",
    "clarification required":              "clarification_required",
    "parse failed":                        "failed",
    "failed":                              "failed",
    "awaiting input":                      "awaiting_input",
}


def _classify_outcome(status: Optional[str]) -> str:
    """Map an agent ``status`` string onto the closed ``outcome`` vocabulary."""
    text = (status or "").strip().lower()
    if text in _OUTCOME_BY_STATUS:
        return _OUTCOME_BY_STATUS[text]
    # Unrecognised status (a new tool, or a variant suffix): fall back to the
    # substrings the existing vocabulary is built from rather than guessing.
    if "no data" in text:
        return "no_data"
    if text.startswith("success"):
        return "success"
    if "fail" in text:
        return "failed"
    return "unknown"


def _tremors_version() -> str:
    """Installed package version, for capability detection by a caller."""
    try:
        from importlib.metadata import version
        return version("tremors")
    except Exception:  # noqa: BLE001 — version is cosmetic; never fail on it
        return "unknown"


# ---------------------------------------------------------------------------
# Output channels
# ---------------------------------------------------------------------------

# Set once in main() from --json. Read by _say/_die so that every code path can
# honour the "stdout carries only the JSON document" rule without threading the
# flag through every signature.
_JSON_MODE = False


def _say(*parts: Any) -> None:
    """Write a human-facing line to **stderr**.

    Never stdout: under ``--json`` stdout carries exactly one JSON document, and
    a stray progress line would make it unparseable. In a terminal both streams
    land on the tty, so interactive output looks the same as before.
    """
    print(*parts, file=sys.stderr)


def _emit_json(document: dict) -> None:
    """Write the single machine-readable document to stdout."""
    # default=str so an unexpected non-serialisable value degrades to its repr
    # instead of raising and denying the caller any result at all.
    json.dump(document, sys.stdout, indent=2, default=str, sort_keys=False)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _die(
    message:  str,
    *,
    code:     int = _EXIT_CONFIG,
    outcome:  str = "config_error",
    **fields: Any,
) -> None:
    """Report a fatal problem and exit with *code*.

    Under ``--json`` the problem is reported as a JSON document too, so a harness
    never has to fall back to parsing prose to find out what went wrong.
    """
    if _JSON_MODE:
        _emit_json({
            "schema_version":  _SCHEMA_VERSION,
            "tremors_version": _tremors_version(),
            "outcome":         outcome,
            "status":          outcome.replace("_", " ").title(),
            "error":           message,
            **fields,
        })
    else:
        _say(message)
    raise SystemExit(code)


# ---------------------------------------------------------------------------
# User configuration file
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = "~/.tremors/config.json"

# Settings that mean the same thing whichever backend runs. They live at the top
# level of the file.
_GLOBAL_CONFIG_KEYS = ("backend", "output_dir")

# Connection settings, stored per backend under ``backends.<name>``. They are
# scoped because they are not interchangeable: a gateway model id means nothing
# to ollama, and a temperature some gateways reject is exactly what ollama
# wants. One flat ``model`` key handed whichever backend ran second a value that
# was never meant for it.
_BACKEND_CONFIG_KEYS = ("model", "base_url", "api_key", "temperature")

# Where the per-backend sections hang.
_CONFIG_BACKENDS_KEY = "backends"

# Everything the file may name. Anything else is a typo (``base-url`` for
# ``base_url``) and is rejected rather than silently ignored — a config the user
# believes is in effect but is not could point a run at the wrong gateway.
_CONFIG_KEYS = _GLOBAL_CONFIG_KEYS + _BACKEND_CONFIG_KEYS

# Never echoed back by `tremors config` or any --json document.
_SECRET_CONFIG_KEYS = frozenset({"api_key"})

# Which environment variable supplies each connection setting, per backend. Used
# both by _build_llm and by `tremors config` when it reports where a value came
# from, so the two can never disagree about precedence.
_ENV_FOR_BACKEND = {
    # ANTHROPIC_AUTH_TOKEN is accepted alongside ANTHROPIC_API_KEY because
    # gateway users commonly hold a bearer token in it rather than a key (it is
    # what the Anthropic SDK reads for `Authorization: Bearer`, and what the LANL
    # AI Portal environment exports). The gateway takes the same credential in
    # the x-api-key header, so it is passed through as api_key.
    "anthropic": {"api_key": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                  "base_url": "ANTHROPIC_BASE_URL",
                  "model":    "ANTHROPIC_MODEL"},
    "openai":    {"api_key": "OPENAI_API_KEY",
                  "base_url": "OPENAI_BASE_URL",
                  "model":    "OPENAI_MODEL"},
    "ollama":    {"base_url": "OLLAMA_BASE_URL",
                  "model":    "OLLAMA_MODEL"},
}

_config_cache: dict = {}


def _config_path(args: Optional[argparse.Namespace] = None) -> Path:
    """Resolve the config file location from flag, env var, then default."""
    raw = (
        getattr(args, "config_file", None)
        or os.environ.get("TREMORS_CONFIG")
        or _DEFAULT_CONFIG_PATH
    )
    return Path(raw).expanduser()


def _load_user_config(args: Optional[argparse.Namespace] = None) -> dict:
    """
    Read ``~/.tremors/config.json`` (override: ``--config``/``TREMORS_CONFIG``).

    This is the lowest-precedence source of settings, below both CLI flags and
    environment variables. It exists so the anthropic backend — which has no
    built-in endpoint or model, because it targets arbitrary
    Anthropic-compatible gateways — can be the default backend without every
    invocation re-supplying three values.

    The file lives outside the repository on purpose: a real key or a private
    endpoint must never land in a tracked file (see the ``.local.`` convention
    in ``.gitignore``). ``tremors config --save`` writes it ``0600``.

    Returns the normalized nested form (see :func:`_normalize_config`)::

        {"backend": …, "output_dir": …, "backends": {"<name>": {…}}}

    Global keys sit at the top level, so ``_setting(…, config, "backend"/…)``
    reads them straight off this dict. Connection settings need
    :func:`_config_view` to select the right backend's section first.

    A malformed file is fatal (exit 5), not ignored: silently falling back to
    defaults could send a query to the wrong service.
    """
    path = _config_path(args)
    cache_key = str(path)
    if cache_key in _config_cache:
        return _config_cache[cache_key]

    config: dict = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _die(f"Could not read the config file {path}: {exc}",
                 config_file=str(path))
        if not isinstance(loaded, dict):
            _die(f"The config file {path} must contain a JSON object.",
                 config_file=str(path))
        config = _normalize_config(loaded, path)

    _config_cache[cache_key] = config
    return config


def _normalize_config(loaded: dict, path: Path) -> dict:
    """Validate a config file's contents and return the nested form.

    Both shapes are accepted and the nested one always comes back, so every
    caller sees a single structure. A **legacy flat file** — connection settings
    at the top level, which is what every file written before per-backend
    sections existed looks like — is migrated *at read time, without rewriting
    the file*: its connection settings are attributed to the backend the file
    itself names, falling back to ``_DEFAULT_BACKEND``. That preserves the
    author's intent (those values were stored while that backend was in effect)
    while keeping them away from the backends they were never meant for. An
    explicit section for the same backend wins over the migrated keys.

    Fails closed on anything it cannot interpret: an unknown top-level key, a
    non-object ``backends``, an unknown backend name, or an unknown key inside a
    section each exit 5 rather than being dropped.
    """
    valid_top = (set(_GLOBAL_CONFIG_KEYS) | {_CONFIG_BACKENDS_KEY}
                 | set(_BACKEND_CONFIG_KEYS))         # legacy flat keys
    unknown = sorted(set(loaded) - valid_top)
    if unknown:
        _die(
            f"Unknown setting(s) in {path}: {', '.join(unknown)}\n"
            f"Valid top-level settings: {', '.join(_GLOBAL_CONFIG_KEYS)}, "
            f"{_CONFIG_BACKENDS_KEY}\n"
            f"Per-backend settings ({', '.join(_BACKEND_CONFIG_KEYS)}) belong "
            f'under "{_CONFIG_BACKENDS_KEY}": {{"<backend>": {{…}}}}',
            config_file=str(path),
            unknown=unknown,
        )

    config = {k: v for k, v in loaded.items()
              if k in _GLOBAL_CONFIG_KEYS and v is not None}

    sections = loaded.get(_CONFIG_BACKENDS_KEY)
    if sections is None:
        sections = {}
    if not isinstance(sections, dict):
        _die(f'"{_CONFIG_BACKENDS_KEY}" in {path} must be a JSON object mapping '
             f"each backend name to its settings.", config_file=str(path))

    backends: dict = {}
    for name, values in sections.items():
        if name not in _BACKENDS:
            _die(f"Unknown backend {name!r} under \"{_CONFIG_BACKENDS_KEY}\" in "
                 f"{path}.\nKnown backends: {', '.join(_BACKENDS)}",
                 config_file=str(path), unknown=[name])
        if not isinstance(values, dict):
            _die(f'"{_CONFIG_BACKENDS_KEY}.{name}" in {path} must be a JSON '
                 f"object.", config_file=str(path))
        bad = sorted(set(values) - set(_BACKEND_CONFIG_KEYS))
        if bad:
            _die(
                f'Unknown setting(s) under "{_CONFIG_BACKENDS_KEY}.{name}" in '
                f"{path}: {', '.join(bad)}\n"
                f"Valid per-backend settings: {', '.join(_BACKEND_CONFIG_KEYS)}",
                config_file=str(path),
                unknown=bad,
            )
        kept = {k: v for k, v in values.items() if v is not None}
        if kept:
            backends[name] = kept

    legacy = {k: v for k, v in loaded.items()
              if k in _BACKEND_CONFIG_KEYS and v is not None}
    if legacy:
        owner = config.get("backend") or _DEFAULT_BACKEND
        migrated = dict(legacy)
        migrated.update(backends.get(owner, {}))
        backends[owner] = migrated

    if backends:
        config[_CONFIG_BACKENDS_KEY] = backends
    return config


def _config_view(config: dict, backend: Optional[str]) -> dict:
    """Flatten *config* to the settings that apply to *backend*.

    :func:`_setting` resolves one flat mapping, so this is where the nested file
    shape collapses into it: global keys first, then the named backend's own
    section on top. A backend with no section sees no stored connection settings
    at all, which is the entire point of the split — the model id stored for a
    gateway is never offered to ollama.
    """
    view = {k: v for k, v in config.items() if k in _GLOBAL_CONFIG_KEYS}
    if backend:
        view.update(config.get(_CONFIG_BACKENDS_KEY, {}).get(backend, {}))
    return view


def _setting(flag_value, env_name, config: dict, config_key: str, default=None):
    """Resolve one setting: CLI flag > environment variable > config file > default.

    *env_name* is one variable name or a tuple of them, tried in order — a
    setting can have more than one conventional home (see ``_ENV_FOR_BACKEND``).

    Returns ``(value, source)``. *source* is what ``tremors config`` reports and
    is why "which model am I actually using" never needs guessing.
    """
    if flag_value is not None:
        return flag_value, "flag"
    for name in ((env_name,) if isinstance(env_name, str) else (env_name or ())):
        from_env = os.environ.get(name)
        if from_env:
            return from_env, f"env:{name}"
    if config.get(config_key) is not None:
        return config[config_key], "config"
    return default, "default" if default is not None else "unset"


def _redact_config(config: dict) -> dict:
    """A copy of *config* with secrets replaced by a presence marker.

    Recurses into the per-backend sections, because that is where a stored key
    actually lives — a redaction that only covered the top level would print
    every one of them.
    """
    redacted = {k: ("<set>" if k in _SECRET_CONFIG_KEYS else v)
                for k, v in config.items() if k != _CONFIG_BACKENDS_KEY}
    sections = config.get(_CONFIG_BACKENDS_KEY)
    if sections:
        redacted[_CONFIG_BACKENDS_KEY] = {
            name: {k: ("<set>" if k in _SECRET_CONFIG_KEYS else v)
                   for k, v in values.items()}
            for name, values in sections.items()
        }
    return redacted


def _build_llm(
    backend:  str,
    model:    Optional[str],
    base_url: Optional[str],
    temperature: Optional[float],
    config:   Optional[dict] = None,
):
    """
    Instantiate the appropriate LangChain chat model for *backend*.

    Exits (code 5, ``config_error``) with a helpful message if a required
    setting or package is missing. *config* is the user config file's contents
    in nested form (see :func:`_load_user_config`); only the section belonging to
    *backend* is consulted, and only below flags and environment variables.

    *temperature* is optional: ``None`` means "let the backend decide". For
    ollama/openai a ``None`` falls back to 0.7 (historical default); for
    anthropic the parameter is sent only when explicitly set, because some
    Anthropic-compatible gateways reject ``temperature`` for certain models
    (e.g. LANL's ``aws-gov.claude-opus-4-8`` returns 400 "temperature is
    deprecated for this model").
    """
    config = _config_view(config or {}, backend)
    env    = _ENV_FOR_BACKEND.get(backend, {})

    def resolve(flag_value, key, default=None):
        return _setting(flag_value, env.get(key), config, key, default)[0]

    resolved_model = resolve(model, "model", _DEFAULT_MODELS.get(backend))

    if backend == "ollama":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            _die(
                "langchain-openai is required for the ollama backend.\n"
                "Install it with:  pip install langchain-openai"
            )
        return ChatOpenAI(
            base_url=resolve(base_url, "base_url", _DEFAULT_OLLAMA_URL),
            api_key="ollama",
            model=resolved_model,
            temperature=temperature if temperature is not None else 0.7,
        )

    if backend == "openai":
        api_key = resolve(None, "api_key")
        if not api_key:
            _die(
                "No OpenAI API key found.\n"
                "Export it:      export OPENAI_API_KEY=sk-...\n"
                "or store it:    tremors config --save --backend openai --api-key sk-..."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            _die(
                "langchain-openai is required for the openai backend.\n"
                "Install it with:  pip install langchain-openai"
            )
        kwargs = dict(
            api_key=api_key,
            model=resolved_model,
            temperature=temperature if temperature is not None else 0.7,
        )
        # Only pass base_url when one was actually supplied, so the official API
        # endpoint stays the library's own default.
        endpoint = resolve(base_url, "base_url")
        if endpoint:
            kwargs["base_url"] = endpoint
        return ChatOpenAI(**kwargs)

    if backend == "anthropic":
        # All three connection fields are required (no built-in defaults): the
        # backend targets arbitrary Anthropic-compatible gateways, so guessing an
        # endpoint or model would silently hit the wrong service. Each may come
        # from a flag, its env var, or the config file — report *every* missing
        # one at once, with both ways to supply it.
        # resolved_model already went through the same flag/env/config search
        # above; _DEFAULT_MODELS has no anthropic entry, so it may still be None.
        api_key  = resolve(None, "api_key")
        base_url = resolve(base_url, "base_url")

        # `missing` in the JSON document carries the bare setting names, the same
        # vocabulary `tremors config` reports, so a harness can branch on it. The
        # prose remediation belongs in `error`, not in a machine-read array.
        wanted = [
            ("api_key",  api_key,        "export ANTHROPIC_API_KEY=…   (ANTHROPIC_AUTH_TOKEN also read)"),
            ("base_url", base_url,       "export ANTHROPIC_BASE_URL=https://…   (or --base-url)"),
            ("model",    resolved_model, "export ANTHROPIC_MODEL=…              (or --model)"),
        ]
        missing  = [name for name, value, _hint in wanted if not value]
        resolved = [name for name, value, _hint in wanted if value]
        if missing:
            # Header, pluralization and the suggested command all follow `missing`.
            # A fixed "all three" above a bullet list that only shows what is
            # absent reads as a bug the moment a stored config has supplied two of
            # them, and telling the user to re-pass --base-url/--model they already
            # have invites them to overwrite good values with placeholders.
            if len(missing) == 3:
                lead = ("anthropic is the default backend. It targets any "
                        "Anthropic-compatible gateway, so it\nhas no built-in "
                        "endpoint or model, and needs all three of these:")
            elif len(missing) == 2:
                lead = "The anthropic backend still needs two settings:"
            else:
                lead = "The anthropic backend is set up except for one setting:"

            lines = [lead]
            lines += [f"  • {name:9s} — {hint}"
                      for name, value, hint in wanted if not value]
            if resolved:
                # Naming what *did* resolve answers "but I configured this" without
                # the user having to go read the file themselves.
                lines += ["",
                          f"({' and '.join(resolved)} resolved fine — run "
                          "`tremors config` to see where each value came from.)"]

            # `--api-key -` rather than a literal key: reading from stdin keeps the
            # secret out of the shell history and the process table.
            save_flag = {"api_key":  "--api-key -",
                         "base_url": "--base-url https://…",
                         "model":    "--model …"}
            flags   = " ".join(save_flag[name] for name in missing)
            pronoun = "it" if len(missing) == 1 else "them"
            command = f"  tremors config --save --backend anthropic {flags}"
            lines += ["",
                      f"Store {pronoun} once instead of exporting {pronoun} "
                      "every session:"]
            lines += ([command] if len(command) <= 76 else
                      ["  tremors config --save --backend anthropic \\",
                       f"                 {flags}"])
            if "api_key" in missing:
                lines += ["  (the trailing `-` reads the key from stdin, so it "
                          "stays out of your shell history)"]
            lines += [f"  (written to {_config_path()} under backends.anthropic, "
                      "mode 0600)",
                      "",
                      "Or switch backends for this run:",
                      "  --backend ollama    # local, no key needed",
                      "  --backend openai    # needs OPENAI_API_KEY"]
            _die(
                "\n".join(lines),
                missing=missing,
                backend=backend,
                config_file=str(_config_path()),
            )

        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            _die(
                "langchain-anthropic is required for the anthropic backend.\n"
                'Install it with:  pip install -e ".[anthropic]"'
            )
        # Only forward temperature when the user explicitly asked for one: some
        # gateways/models reject the parameter outright (400), so an unrequested
        # default must not be imposed.
        kwargs = dict(api_key=api_key, base_url=base_url, model=resolved_model)
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)

    _die(f"Unknown backend: {backend!r}. Choose from: ollama, openai, anthropic")


def _scrub_secrets(text: str) -> str:
    """Redact anything that looks like an API key from *text*.

    A bad-key 401 from some gateways echoes a fragment of the submitted key in
    its error body; never surface that to the terminal.
    """
    return re.sub(r"sk-[A-Za-z0-9_\-]+", "sk-***", text)


def _preflight_check(llm, backend: str) -> None:
    """Send one cheap request to *llm* to confirm the backend is reachable.

    Runs before the real query so a bad key / endpoint / model fails fast with a
    clear, secret-safe message instead of surfacing mid-pipeline. Classifies the
    failure by exception *class* (never by dumping the raw message, which may
    contain a key fragment) and exits with remediation. Skipped via
    ``--no-preflight``.
    """
    try:
        # A plain string becomes a non-empty HumanMessage, so this is a valid
        # request for every backend (and avoids the empty-`messages` 400 that a
        # system-only prompt triggers on Anthropic).
        llm.invoke("ping")
    except Exception as exc:  # noqa: BLE001 — map any provider error to advice
        name = type(exc).__name__.lower()

        if "auth" in name or "permission" in name or "forbidden" in name:
            detail = (
                "authentication failed — the credential was rejected.\n"
                "anthropic reads ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN, then\n"
                "the stored api_key; openai reads OPENAI_API_KEY. Run `tremors config`\n"
                "to see which source is supplying the one being used — a stale stored\n"
                "key is used only when no environment variable is set."
            )
        elif "notfound" in name:
            detail = (
                "the model was not found on this endpoint.\n"
                "Check --model / ANTHROPIC_MODEL matches a model the backend serves."
            )
        elif "connection" in name or "timeout" in name:
            detail = (
                "could not connect to the endpoint.\n"
                "Check the base URL (--base-url / ANTHROPIC_BASE_URL) and your network."
            )
        else:
            detail = _scrub_secrets(str(exc)) or type(exc).__name__

        _die(
            f"Preflight check failed for the {backend} backend: {detail}\n"
            "(pass --no-preflight to skip this check)",
            reason=type(exc).__name__,
        )


# ---------------------------------------------------------------------------
# Durable sessions
# ---------------------------------------------------------------------------

_DEFAULT_SESSION_DIR = "~/.tremors"


class _SessionStore:
    """
    On-disk home for resumable runs.

    Two files live side by side in the session directory:

    ``sessions.sqlite``
        The LangGraph ``SqliteSaver`` checkpoints. This is what actually lets a
        paused graph be picked up by a different process — the default
        ``InMemorySaver`` cannot, since it dies with the interpreter.
    ``sessions.json``
        A small index mapping session id → the invocation that created it
        (backend, model, base URL, temperature, output dir, interrupt mode) plus
        the pending gate payload. It exists so ``tremors resume`` needs nothing
        but the session id: rebuilding the agent requires the same LLM
        configuration, and a harness should not have to remember and re-supply
        it.

    The index never stores an API key. Credentials come from the environment on
    every invocation, so a session file is safe to leave on disk.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path    = self.directory / "sessions.sqlite"
        self.index_path = self.directory / "sessions.json"
        self._conn: Optional[sqlite3.Connection] = None
        self._saver: Any = None

    @property
    def saver(self):
        """The durable checkpointer, created on first use."""
        if self._saver is None:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
            except ImportError:
                _die(
                    "Durable sessions (--session-id) need the SQLite checkpointer.\n"
                    "Install it with:  pip install langgraph-checkpoint-sqlite"
                )
            # check_same_thread=False: the graph and its middleware may touch the
            # checkpointer from a worker thread, and sqlite3 otherwise refuses.
            self._conn  = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._saver = SqliteSaver(self._conn)
        return self._saver

    # ── index ────────────────────────────────────────────────────────
    def _read_index(self) -> dict:
        if not self.index_path.is_file():
            return {}
        try:
            data = json.loads(self.index_path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            # A corrupt index must not strand a run; the checkpoints are the
            # source of truth for graph state, this file only caches config.
            logger.warning("Session index at %s is unreadable; ignoring.", self.index_path)
            return {}

    def _write_index(self, index: dict) -> None:
        # Write-then-rename so a crash mid-write cannot truncate the index.
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, indent=2, default=str))
        tmp.replace(self.index_path)

    def load(self, session_id: str) -> Optional[dict]:
        return self._read_index().get(session_id)

    def save(self, session_id: str, record: dict) -> None:
        index = self._read_index()
        index[session_id] = {**index.get(session_id, {}), **record}
        self._write_index(index)

    def list_sessions(self) -> dict:
        return self._read_index()

    def drop(self, session_id: str) -> bool:
        index = self._read_index()
        existed = index.pop(session_id, None) is not None
        if existed:
            self._write_index(index)
        return existed


def _session_store(args: argparse.Namespace) -> _SessionStore:
    """Resolve the session directory from flag, env var, then default."""
    directory = (
        getattr(args, "session_dir", None)
        or os.environ.get("TREMORS_SESSION_DIR")
        or _DEFAULT_SESSION_DIR
    )
    return _SessionStore(Path(directory).expanduser().resolve())


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """The directory the agent writes to: flag > env > config file > default.

    Every caller must go through this, so the path reported in the JSON, recorded
    in a session, and used by the agent cannot drift apart. In the ``resume``
    path ``args.output_dir`` already carries the session's recorded directory,
    which is how a record outranks the ambient environment.
    """
    raw = _setting(getattr(args, "output_dir", None), "TREMORS_OUTPUT_DIR",
                   _load_user_config(args), "output_dir", _DEFAULT_OUTPUT_DIR)[0]
    return Path(raw).expanduser().resolve()


def _build_agent(args: argparse.Namespace, checkpointer: Any = None):
    """Build the TremorsAgent from parsed CLI args."""
    try:
        from tremors import TremorsAgent
    except ImportError as exc:
        _die(f"Could not import TremorsAgent: {exc}")

    config  = _load_user_config(args)
    backend = _setting(getattr(args, "backend", None), "TREMORS_BACKEND",
                       config, "backend", _DEFAULT_BACKEND)[0]

    llm = _build_llm(
        backend=backend,
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
        temperature=_setting(args.temperature, None, config, "temperature")[0],
        config=config,
    )

    if not getattr(args, "no_preflight", False):
        _preflight_check(llm, backend)

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    return TremorsAgent(
        llm=llm,
        output_dir=str(output_dir),
        checkpointer=checkpointer,
        interrupts=not getattr(args, "no_interrupts", False),
    )


# ---------------------------------------------------------------------------
# Human-in-the-loop gates
# ---------------------------------------------------------------------------

class _Aborted(Exception):
    """Raised when the user declines a gate outright, to unwind out of the run."""


class _Unanswerable(Exception):
    """Raised when ``--yes`` meets a gate it is not allowed to answer.

    Only a clarification qualifies: approving a plan the verifier called unusable
    is not a decision a flag gets to make. The run stays paused in the
    checkpointer, so a durable session can still be resumed with ``--clarify``.
    """

    def __init__(self, payload: dict):
        super().__init__("clarification cannot be auto-answered")
        self.payload = payload


def _prompt(question: str) -> str:
    """Read one line of input, treating EOF as an abort rather than an empty answer.

    The question goes to stderr (not through ``input``'s own stdout prompt) to
    keep stdout reserved for the JSON document.
    """
    sys.stderr.write(question)
    sys.stderr.flush()
    try:
        return input().strip()
    except EOFError:
        raise _Aborted("input stream closed")


def _auto_approve_gate(payload: dict) -> dict:
    """``--yes``: approve every approval gate, refuse to invent a clarification."""
    if payload.get("kind") == "clarification":
        raise _Unanswerable(payload)

    requests = payload.get("action_requests") or []
    for request in requests:
        _say(f"\n[tremors] auto-approving {request.get('name')} (--yes)")
    # One decision per pending call: the middleware raises ValueError otherwise.
    return {"decisions": [{"type": "approve"} for _ in requests] or [{"type": "approve"}]}


def _gate_prompt(payload: dict) -> object:
    """
    Answer one human-in-the-loop request from the terminal.

    Two kinds of request arrive here, and they take different resume values:

    * a **clarification** pause raised inside ``plan_query``, whose answer is
      free text that the query is re-planned with;
    * an **approval gate** from ``HumanInTheLoopMiddleware``, whose answer is
      ``{"decisions": [...]}`` with exactly one decision per pending tool call.

    Only ever called with a terminal attached — see :func:`_gate_callback`, which
    hands back ``None`` instead when there is no tty so that the run surfaces the
    pending request as data rather than blocking on stdin.
    """
    kind = payload.get("kind")

    if kind == "clarification":
        problems = payload.get("problems") or []
        _say("\n" + "═" * 60)
        _say("  Clarification needed — these search parameters can't be used:")
        for problem in problems:
            _say(f"    • {problem}")
        _say("═" * 60)

        answer = _prompt("\nClarify (or press Ctrl-D to abort): ")
        if not answer:
            raise _Aborted("no clarification given")
        return answer

    # ── Approval gate ─────────────────────────────────────────────────
    requests = payload.get("action_requests") or []
    decisions = []

    for request in requests:
        _say("\n" + "═" * 60)
        _say(f"  Approval needed: {request.get('name')}")
        _say("═" * 60)
        _say(request.get("description") or "(no description)")

        while True:
            choice = _prompt("\n[a]pprove  [e]dit parameters  [r]eject  [q]uit: ").lower()

            if choice in ("a", "approve", ""):
                decisions.append({"type": "approve"})
                break

            if choice in ("e", "edit"):
                _say(
                    'Enter a JSON object of parameters to override, e.g.\n'
                    '  {"min_mag": 6.0, "max_date": "2020-06-30"}'
                )
                raw = _prompt("JSON: ")
                try:
                    override = json.loads(raw)
                except json.JSONDecodeError as exc:
                    _say(f"  Not valid JSON ({exc}); try again.")
                    continue
                if not isinstance(override, dict):
                    _say("  Expected a JSON object (key/value pairs); try again.")
                    continue
                decisions.append({
                    "type": "edit",
                    "edited_action": {
                        "name": request.get("name"),
                        # Merge over the original args so nothing else is dropped.
                        "args": {**(request.get("args") or {}), "params_override": override},
                    },
                })
                break

            if choice in ("r", "reject"):
                reason = _prompt("Reason (sent back to the model, which will re-plan): ")
                decisions.append({"type": "reject", "message": reason or "rejected by user"})
                break

            if choice in ("q", "quit", "abort"):
                raise _Aborted("rejected at the approval gate")

            _say("  Please answer a, e, r, or q.")

    return {"decisions": decisions}


def _gate_callback(args: argparse.Namespace):
    """
    Choose how gates get answered for this invocation.

    * ``--yes`` → approve approval gates automatically.
    * a terminal → ask the user.
    * neither → ``None``, which makes ``agent.run`` return
      ``status="Awaiting Input"`` with the payload attached instead of blocking
      on stdin. That is what turns an unattended gate into exit 4 plus a machine
      -readable ``interrupt`` the caller can act on, rather than a hang.
    """
    if args.yes:
        return _auto_approve_gate
    if sys.stdin.isatty():
        return _gate_prompt
    return None


# ---------------------------------------------------------------------------
# Result projection
# ---------------------------------------------------------------------------

_ARTIFACT_KEYS = (
    "plots",
    "waveforms_saved",
    "waveform_plots",
    "continuous_waveforms_saved",
    "continuous_waveform_plots",
)


def _result_document(
    result:     dict,
    *,
    query:      Optional[str],
    output_dir: Optional[str],
    session_id: Optional[str] = None,
    resumable:  bool = False,
    extra:      Optional[dict] = None,
) -> dict:
    """
    Build the ``--json`` document from an agent result dict.

    Artifact lists are **complete** here, unlike the human summary, which
    truncates them: a caller that has to re-glob the output directory to find out
    what was written cannot be said to be driving the tool.
    """
    pending = result.get("interrupt")
    outcome = "awaiting_input" if pending else _classify_outcome(result.get("status"))

    document = {
        "schema_version":  _SCHEMA_VERSION,
        "tremors_version": _tremors_version(),
        "outcome":         outcome,
        "status":          result.get("status", "Unknown"),
        "error":           result.get("error"),
        "problems":        list(result.get("problems") or []),
        "query":           query,
        "output_dir":      output_dir,
        "session_id":      session_id,
        "resumable":       bool(resumable and pending),
        "search_params":   result.get("search_params") or {},
        "datacenter":      result.get("datacenter"),
        "queried_dcs":     list(result.get("queried_dcs") or []),
        "metadata_tables": dict(result.get("metadata_tables") or {}),
    }
    for key in _ARTIFACT_KEYS:
        document[key] = list(result.get(key) or [])

    document["counts"] = {
        "metadata_tables": len(document["metadata_tables"]),
        **{key: len(document[key]) for key in _ARTIFACT_KEYS},
    }

    if pending:
        document["interrupt"] = pending
    if extra:
        document.update(extra)
    return document


def _print_result(result: dict) -> None:
    """Print a human-readable summary of an agent result dict (to stderr)."""
    status = result.get("status", "Unknown")
    error  = result.get("error")

    _say(f"\n{'─' * 60}")
    _say(f"  Status : {status}")

    if error:
        _say(f"  Error  : {error}")

    pending = result.get("interrupt")
    if pending:
        names = ", ".join(
            str(r.get("name")) for r in (pending.get("action_requests") or [])
        )
        kind = pending.get("kind") or "approval"
        _say(f"  Paused : awaiting {kind}{f' for {names}' if names else ''}")

    queried = result.get("queried_dcs") or []
    if queried:
        _say(f"  DCs    : {', '.join(queried)}")

    tables = result.get("metadata_tables") or {}
    if tables:
        _say(f"  Tables : {', '.join(tables.keys())}")
        for name, path in tables.items():
            _say(f"           {name:20s} → {path}")

    plots = result.get("plots") or []
    for p in plots:
        _say(f"  Plot   : {p}")

    waveforms = result.get("waveforms_saved") or result.get("continuous_waveforms_saved") or []
    if waveforms:
        _say(f"  Waveforms saved : {len(waveforms)} file(s)")
        for w in waveforms[:5]:
            _say(f"           {w}")
        if len(waveforms) > 5:
            _say(f"           … and {len(waveforms) - 5} more")

    wplots = result.get("waveform_plots") or result.get("continuous_waveform_plots") or []
    for wp in wplots:
        _say(f"  Waveform plot : {wp}")

    _say(f"{'─' * 60}\n")


def _finish(document: dict) -> None:
    """Emit the result (JSON if asked) and exit with the code for its outcome."""
    if _JSON_MODE:
        _emit_json(document)
    raise SystemExit(_EXIT_FOR_OUTCOME.get(document.get("outcome"), _EXIT_FAILED))


def _resume_hint(session_id: Optional[str], pending: Optional[dict]) -> None:
    """Tell a human (or a log reader) how to continue a paused session."""
    if not pending:
        return
    if not session_id:
        _say(
            "  Hint   : this pause cannot be resumed — no --session-id was given, and "
            "the\n           in-memory checkpointer does not outlive the process. "
            "Re-run with\n           --session-id NAME for per-gate control, or -y to "
            "auto-approve."
        )
        return
    if pending.get("kind") == "clarification":
        _say(f'  Resume : tremors resume --session-id {session_id} --clarify "…"')
    else:
        _say(f"  Resume : tremors resume --session-id {session_id} --approve")
        _say(f'           tremors resume --session-id {session_id} --reject "why"')


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_query(args: argparse.Namespace) -> None:
    """Run a single natural-language query and exit."""
    session    = _session_store(args) if args.session_id else None
    output_dir = str(_resolve_output_dir(args))

    agent = _build_agent(args, checkpointer=session.saver if session else None)

    # ── Plan-only dry run ─────────────────────────────────────────────
    # Resolves the query and stops. No FDSN request, no files, no gates.
    if args.plan_only:
        _say(f"[tremors] Planning (dry run, no FDSN): {args.query!r}")
        plan = agent.plan(args.query)
        document = _result_document(
            plan,
            query=args.query,
            output_dir=output_dir,
            extra={
                "plan_only":          True,
                "target_datacenters": plan.get("target_datacenters") or [],
            },
        )
        _print_result(plan)
        if document["outcome"] == "success":
            targets = ", ".join(document["target_datacenters"]) or "(none)"
            _say(f"  Would query : {targets}")
            _say(f"{'─' * 60}\n")
        _finish(document)

    if session:
        # Persist the invocation *before* running, so a session that pauses (or
        # crashes) is still resumable without the caller re-supplying the config.
        session.save(args.session_id, {
            "query":         args.query,
            "output_dir":    output_dir,
            "backend":       args.backend,
            "model":         args.model,
            "base_url":      args.base_url,
            "temperature":   args.temperature,
            "no_interrupts": args.no_interrupts,
            "created":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    _say(f"[tremors] Running query: {args.query!r}")

    try:
        result = agent.run(
            args.query,
            on_interrupt=_gate_callback(args),
            thread_id=args.session_id or None,
        )
    except _Aborted as exc:
        _die(
            f"[tremors] Aborted: {exc}. No further data was retrieved.",
            code=_EXIT_ABORTED,
            outcome="aborted",
        )
    except _Unanswerable as exc:
        _finish(_unanswerable_document(exc.payload, args, session, output_dir))

    _report(result, args, session, output_dir)


def _unanswerable_document(
    payload:    dict,
    args:       argparse.Namespace,
    session:    Optional[_SessionStore],
    output_dir: str,
) -> dict:
    """Document for a clarification that ``--yes`` refused to answer.

    Resumable sessions report ``awaiting_input`` (exit 4) because ``--clarify``
    genuinely can continue them; otherwise the pause is terminal for this process
    and ``clarification_required`` (exit 3) is the honest answer.
    """
    problems = list(payload.get("problems") or [])
    _say("\n" + "═" * 60)
    _say("  Clarification needed — these search parameters can't be used:")
    for problem in problems:
        _say(f"    • {problem}")
    _say("═" * 60)
    _say("  --yes cannot answer a clarification: approving a plan the verifier")
    _say("  rejected is not a decision a flag gets to make.")

    if session:
        session.save(args.session_id, {"pending": payload, "outcome": "awaiting_input"})
        _resume_hint(args.session_id, payload)
        return _result_document(
            {
                "status":        "Awaiting Input",
                "error":         " ".join(problems) or None,
                "problems":      problems,
                "search_params": payload.get("params") or {},
                "interrupt":     payload,
            },
            query=args.query,
            output_dir=output_dir,
            session_id=args.session_id,
            resumable=True,
        )

    _say(
        "  Re-run with the missing detail in the query (e.g. add a search radius),\n"
        "  or with --session-id NAME so the pause can be answered by\n"
        "  `tremors resume --clarify`."
    )
    return _result_document(
        {
            "status":        "Clarification Required",
            "error":         " ".join(problems) or None,
            "problems":      problems,
            "search_params": payload.get("params") or {},
        },
        query=args.query,
        output_dir=output_dir,
    )


def _report(
    result:     dict,
    args:       argparse.Namespace,
    session:    Optional[_SessionStore],
    output_dir: str,
    query:      Optional[str] = None,
) -> None:
    """Summarise, persist session state, emit, exit. Shared by query and resume."""
    document = _result_document(
        result,
        query=query if query is not None else getattr(args, "query", None),
        output_dir=output_dir,
        session_id=getattr(args, "session_id", None),
        resumable=bool(session),
    )

    _print_result(result)

    pending = result.get("interrupt")
    if session:
        # Record the pending payload so `resume --show` can re-read it and
        # `resume` can validate a decision against the gate that is actually
        # waiting. Cleared when the run completes.
        session.save(args.session_id, {
            "pending": pending,
            "outcome": document["outcome"],
        })
        _resume_hint(args.session_id, pending)
    elif pending:
        _resume_hint(None, pending)

    _finish(document)


def _build_resume_value(pending: dict, args: argparse.Namespace) -> Any:
    """
    Turn the resume flags into the value the paused graph expects.

    An approval gate takes ``{"decisions": [...]}`` with exactly one decision per
    pending call; a clarification takes the answer text itself. Handing the wrong
    shape to either would be accepted and then misbehave — a clarification would
    receive the decisions dict as its answer text — so the kind is checked here
    rather than left to fail deep in the middleware.
    """
    kind  = pending.get("kind")
    count = len(pending.get("action_requests") or []) or 1
    is_clarification = kind == "clarification"

    def _reject_wrong_kind(flag: str) -> None:
        if is_clarification:
            _die(
                f"This session is waiting for a clarification, which {flag} cannot "
                'answer.\nUse:  --clarify "the missing detail"',
                code=_EXIT_USAGE,
                outcome="config_error",
            )

    if args.clarify is not None:
        if not is_clarification:
            _die(
                "--clarify answers a clarification pause, but this session is waiting "
                "on an approval gate.\nUse --approve, --reject REASON, or --decision JSON.",
                code=_EXIT_USAGE,
                outcome="config_error",
            )
        if not args.clarify.strip():
            _die("--clarify needs non-empty text.", code=_EXIT_USAGE, outcome="config_error")
        return args.clarify

    if args.approve:
        _reject_wrong_kind("--approve")
        return {"decisions": [{"type": "approve"} for _ in range(count)]}

    if args.reject is not None:
        _reject_wrong_kind("--reject")
        message = args.reject or "rejected by user"
        return {"decisions": [{"type": "reject", "message": message} for _ in range(count)]}

    if args.decision is not None:
        _reject_wrong_kind("--decision")
        try:
            parsed = json.loads(args.decision)
        except json.JSONDecodeError as exc:
            _die(f"--decision is not valid JSON: {exc}", code=_EXIT_USAGE, outcome="config_error")

        if isinstance(parsed, dict) and "decisions" in parsed:
            decisions = parsed["decisions"]
        elif isinstance(parsed, dict) and "type" in parsed:
            decisions = [parsed]
        elif isinstance(parsed, list):
            decisions = parsed
        else:
            _die(
                '--decision must be a decision object ({"type": "approve"}), a list of '
                'them, or {"decisions": [...]}.',
                code=_EXIT_USAGE,
                outcome="config_error",
            )

        if not isinstance(decisions, list) or len(decisions) != count:
            _die(
                f"This session has {count} pending call(s) but --decision supplied "
                f"{len(decisions) if isinstance(decisions, list) else 'a non-list'}. "
                "Supply exactly one decision per pending call.",
                code=_EXIT_USAGE,
                outcome="config_error",
            )
        return {"decisions": decisions}

    _die(
        "Nothing to resume with. Pass one of:\n"
        "  --approve                 approve the pending call(s)\n"
        '  --reject "reason"         reject, and let the model re-plan\n'
        '  --decision JSON           a full decision, e.g. an edit\n'
        '  --clarify "text"          answer a clarification pause\n'
        "  --show                    just print what is pending",
        code=_EXIT_USAGE,
        outcome="config_error",
    )


def _cmd_resume(args: argparse.Namespace) -> None:
    """Answer a gate that a previous invocation left pending."""
    session = _session_store(args)
    record  = session.load(args.session_id)

    if record is None:
        known = ", ".join(sorted(session.list_sessions())) or "(none)"
        _die(
            f"No session named {args.session_id!r} in {session.directory}.\n"
            f"Known sessions: {known}",
            code=_EXIT_USAGE,
            outcome="config_error",
        )

    pending = record.get("pending")
    if not pending:
        _die(
            f"Session {args.session_id!r} has nothing pending "
            f"(last outcome: {record.get('outcome', 'unknown')}).\n"
            "Only a run that stopped at a gate can be resumed.",
            code=_EXIT_USAGE,
            outcome="config_error",
        )

    # ── --show: report what is pending and stop ───────────────────────
    if args.show:
        _say(f"[tremors] Session {args.session_id!r} is awaiting "
             f"{pending.get('kind') or 'approval'}.")
        _print_result({"status": "Awaiting Input", "interrupt": pending,
                       "problems": pending.get("problems") or []})
        _resume_hint(args.session_id, pending)
        _finish(_result_document(
            {
                "status":        "Awaiting Input",
                "problems":      list(pending.get("problems") or []),
                "search_params": record.get("search_params") or pending.get("params") or {},
                "interrupt":     pending,
            },
            query=record.get("query"),
            output_dir=record.get("output_dir"),
            session_id=args.session_id,
            resumable=True,
        ))

    value = _build_resume_value(pending, args)

    # Rebuild the agent with the configuration the session was created under,
    # letting an explicit flag on *this* invocation win. Anything neither the
    # flag nor the record supplies is left None so _build_agent applies the
    # normal env > config-file > default search: the session record sits between
    # the two, because resuming under a different backend than the one that
    # paused would answer the gate against a different model.
    output_dir = args.output_dir or record.get("output_dir")
    rebuilt = argparse.Namespace(
        # _unanswerable_document and _report both read .query off this namespace.
        query=record.get("query"),
        backend=args.backend or record.get("backend"),
        model=args.model or record.get("model"),
        base_url=args.base_url or record.get("base_url"),
        temperature=args.temperature if args.temperature is not None else record.get("temperature"),
        no_preflight=args.no_preflight,
        output_dir=output_dir,
        no_interrupts=bool(record.get("no_interrupts", False)),
        yes=args.yes,
        session_id=args.session_id,
        session_dir=args.session_dir,
        config_file=getattr(args, "config_file", None),
    )

    agent      = _build_agent(rebuilt, checkpointer=session.saver)
    output_dir = str(_resolve_output_dir(rebuilt))

    _say(f"[tremors] Resuming session {args.session_id!r} "
         f"({pending.get('kind') or 'approval'}).")

    try:
        result = agent.resume(
            value,
            on_interrupt=_gate_callback(rebuilt),
            thread_id=args.session_id,
        )
    except _Aborted as exc:
        _die(
            f"[tremors] Aborted: {exc}. No further data was retrieved.",
            code=_EXIT_ABORTED,
            outcome="aborted",
        )
    except _Unanswerable as exc:
        _finish(_unanswerable_document(exc.payload, rebuilt, session, output_dir))

    _report(result, rebuilt, session, output_dir, query=record.get("query"))


def _cmd_sessions(args: argparse.Namespace) -> None:
    """List or delete durable sessions."""
    session = _session_store(args)

    if args.delete:
        existed = session.drop(args.delete)
        if not existed:
            _die(f"No session named {args.delete!r} to delete.",
                 code=_EXIT_USAGE, outcome="config_error")
        _say(f"[tremors] Deleted session {args.delete!r}.")
        _finish({
            "schema_version":  _SCHEMA_VERSION,
            "tremors_version": _tremors_version(),
            "outcome":         "success",
            "status":          "Deleted",
            "session_id":      args.delete,
        })

    index = session.list_sessions()
    _say(f"[tremors] {len(index)} session(s) in {session.directory}")
    for sid, record in sorted(index.items()):
        pending = record.get("pending")
        state   = (f"awaiting {pending.get('kind') or 'approval'}" if pending
                   else record.get("outcome", "unknown"))
        _say(f"  {sid:24s} {state:24s} {record.get('query') or ''}")

    _finish({
        "schema_version":  _SCHEMA_VERSION,
        "tremors_version": _tremors_version(),
        "outcome":         "success",
        "status":          "Listed",
        "session_dir":     str(session.directory),
        "sessions": [
            {
                "session_id": sid,
                "query":      record.get("query"),
                "outcome":    record.get("outcome"),
                "created":    record.get("created"),
                "output_dir": record.get("output_dir"),
                "pending_kind": (record.get("pending") or {}).get("kind"),
                "resumable":  bool(record.get("pending")),
            }
            for sid, record in sorted(index.items())
        ],
    })


def _cmd_config(args: argparse.Namespace) -> None:
    """Show or write the stored connection settings.

    Bare ``tremors config`` reports the effective value of every setting and
    where it came from, so "why is it using that model" is answerable without
    guessing at the environment. ``--save`` persists the settings given on the
    same command line; ``--unset`` removes them.

    Connection settings are read and written **per backend**: whichever backend
    is in effect for this command (``--backend`` > ``$TREMORS_BACKEND`` > the
    stored default) names the section they go in or come out of.
    """
    path = _config_path(args)

    # ── writes ────────────────────────────────────────────────────────
    if args.save or args.unset:
        stored = {}
        if path.is_file():
            # Validates the existing file, and normalizes a legacy flat one, so
            # a write never silently drops settings it could not parse.
            stored = copy.deepcopy(_load_user_config(args))

        # The section that per-backend settings are written to or removed from.
        # Resolved before any mutation so `--save --backend X` targets X.
        previous_backend = _setting(None, "TREMORS_BACKEND", stored, "backend",
                                    _DEFAULT_BACKEND)[0]
        target = _setting(args.backend, "TREMORS_BACKEND", stored, "backend",
                          _DEFAULT_BACKEND)[0]
        sections = stored.setdefault(_CONFIG_BACKENDS_KEY, {})

        if args.unset:
            bad = sorted(set(args.unset) - set(_CONFIG_KEYS))
            if bad:
                _die(f"Cannot unset unknown setting(s): {', '.join(bad)}\n"
                     f"Valid settings: {', '.join(_CONFIG_KEYS)}",
                     code=_EXIT_USAGE)
            for key in args.unset:
                if key in _GLOBAL_CONFIG_KEYS:
                    stored.pop(key, None)
                else:
                    sections.get(target, {}).pop(key, None)

        given: dict = {}
        if args.save:
            api_key = args.api_key
            if api_key == "-":
                # Reading from stdin keeps the key out of shell history and out
                # of the process table, where a --api-key argument is visible to
                # every other user on the machine.
                api_key = sys.stdin.readline().strip()
                if not api_key:
                    _die("No API key arrived on stdin.", code=_EXIT_USAGE)
            incoming = {
                "backend":     args.backend,
                "output_dir":  args.output_dir,
                "model":       args.model,
                "base_url":    args.base_url,
                "api_key":     api_key,
                "temperature": args.temperature,
            }
            given = {k: v for k, v in incoming.items() if v is not None}
            if not given:
                _die(
                    "--save needs at least one setting to store, e.g.\n"
                    "  tremors config --save --base-url https://… --model … --api-key sk-…",
                    code=_EXIT_USAGE,
                )
            for key, value in given.items():
                if key in _GLOBAL_CONFIG_KEYS:
                    stored[key] = value
                else:
                    sections.setdefault(target, {})[key] = value

        # Drop sections emptied by --unset so the file does not accumulate
        # `"ollama": {}` husks, and the wrapper too once nothing is left in it.
        for name in [n for n, values in sections.items() if not values]:
            del sections[name]
        if not sections:
            del stored[_CONFIG_BACKENDS_KEY]

        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, and create the temp file already private: a key must
        # never exist on disk world-readable, even briefly.
        tmp = path.with_suffix(".json.tmp")
        tmp.touch(mode=0o600, exist_ok=True)
        tmp.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
        os.chmod(tmp, 0o600)
        tmp.replace(path)
        _config_cache.pop(str(path), None)

        _say(f"[tremors] Wrote {path} (mode 0600)")
        for key in _GLOBAL_CONFIG_KEYS:
            if key in stored:
                _say(f"    {key:12s} {stored[key]}")
        for name in sorted(stored.get(_CONFIG_BACKENDS_KEY, {})):
            _say(f"    {_CONFIG_BACKENDS_KEY}.{name}:")
            values = stored[_CONFIG_BACKENDS_KEY][name]
            for key in _BACKEND_CONFIG_KEYS:
                if key in values:
                    shown = "<set>" if key in _SECRET_CONFIG_KEYS else values[key]
                    _say(f"        {key:12s} {shown}")

        # Say which section was touched, and say it loudly when the default
        # backend moved: a silent default switch is how a run ends up pointed at
        # a service the user did not mean to reach.
        touched = set(given) | set(args.unset or ())
        if touched & set(_BACKEND_CONFIG_KEYS):
            _say(f"[tremors] Connection settings apply to backend {target!r} "
                 f"({_CONFIG_BACKENDS_KEY}.{target}).")
        if args.save and args.backend and args.backend != previous_backend:
            _say(f"[tremors] Default backend is now {target!r} "
                 f"(was {previous_backend!r}) — runs without --backend use it.")

        _finish({
            "schema_version":  _SCHEMA_VERSION,
            "tremors_version": _tremors_version(),
            "outcome":         "success",
            "status":          "Saved",
            "config_file":     str(path),
            "config":          _redact_config(stored),
        })

    # ── reads ─────────────────────────────────────────────────────────
    config  = _load_user_config(args)
    backend = _setting(args.backend, "TREMORS_BACKEND", config, "backend",
                       _DEFAULT_BACKEND)[0]
    # Only this backend's stored section participates in the resolution below,
    # which is what `--backend ollama` has to show to be honest about it.
    view    = _config_view(config, backend)
    env     = _ENV_FOR_BACKEND.get(backend, {})

    # (label, flag value, env var, config key, default) for the effective view.
    rows = [
        ("backend",     args.backend,     "TREMORS_BACKEND",    "backend",
         _DEFAULT_BACKEND),
        ("model",       args.model,       env.get("model"),     "model",
         _DEFAULT_MODELS.get(backend)),
        ("base_url",    args.base_url,    env.get("base_url"),  "base_url",
         _DEFAULT_OLLAMA_URL if backend == "ollama" else None),
        ("api_key",     None,             env.get("api_key"),   "api_key", None),
        ("temperature", args.temperature, None,                 "temperature", None),
        ("output_dir",  args.output_dir,  "TREMORS_OUTPUT_DIR", "output_dir",
         _DEFAULT_OUTPUT_DIR),
    ]

    effective = {}
    _say(f"[tremors] Config file: {path}"
         f"{'' if path.is_file() else '  (does not exist yet)'}")
    _say(f"[tremors] Effective settings for backend {backend!r}"
         " (flag > env > config file > default):")
    for label, flag_value, env_name, config_key, default in rows:
        value, source = _setting(flag_value, env_name, view, config_key, default)
        effective[label] = {
            "value":  "<set>" if (label in _SECRET_CONFIG_KEYS and value) else value,
            "source": source,
        }
        shown = ("<set>" if label in _SECRET_CONFIG_KEYS and value
                 else "—" if value is None else value)
        _say(f"    {label:12s} {str(shown):40s} [{source}]")

    # What each backend cannot run without. anthropic needs all three because it
    # targets arbitrary gateways; ollama needs nothing (local, keyless, defaulted).
    required = {
        "anthropic": ("api_key", "base_url", "model"),
        "openai":    ("api_key", "model"),
        "ollama":    (),
    }.get(backend, ())
    missing = [key for key in required if effective[key]["value"] is None]

    # Sections for the backends this command is not resolving. Shown so that
    # "the file has a model in it, why is mine [default]" answers itself: those
    # values belong to another backend and are deliberately not offered here.
    others = {name: values
              for name, values in config.get(_CONFIG_BACKENDS_KEY, {}).items()
              if name != backend}
    if others:
        _say("")
        _say("[tremors] Also stored, for other backends (not used above):")
        for name in sorted(others):
            keys = ", ".join(
                f"{k}=<set>" if k in _SECRET_CONFIG_KEYS else f"{k}={others[name][k]}"
                for k in _BACKEND_CONFIG_KEYS if k in others[name]
            )
            _say(f"    {_CONFIG_BACKENDS_KEY}.{name:10s} {keys}")

    if missing:
        _say("")
        _say(f"[tremors] Backend {backend!r} cannot run yet — missing: "
             f"{', '.join(missing)}")
        _say(f"          Store them:  tremors config --save --backend {backend} "
             "--base-url https://… --model … --api-key sk-…")
        _say("          Or switch:   --backend ollama   (local, no key needed)")

    _finish({
        "schema_version":  _SCHEMA_VERSION,
        "tremors_version": _tremors_version(),
        "outcome":         "success",
        "status":          "Listed",
        "config_file":     str(path),
        "config_file_exists": path.is_file(),
        "backend":         backend,
        "settings":        effective,
        "config":          _redact_config(config),
        "missing":         missing,
        "ready":           not missing,
    })


def _build_parser() -> argparse.ArgumentParser:
    # ── Top-level parser ──────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="tremors",
        description=(
            "Tremors – FDSN seismic data retrieval and visualization agent.\n\n"
            "First run: point it at a model backend (default: anthropic).\n"
            "  tremors config --save --base-url https://gateway.example \\\n"
            "                 --model my-model-id --api-key sk-...\n"
            "  tremors config                            # check what is in effect\n"
            "\n"
            "Examples:\n"
            "  tremors query \"M5+ earthquakes near Japan in 2020\"\n"
            "  tremors query \"continuous BH* waveforms for CI network, Feb 2016\" "
            "--output-dir ./ci_feb16 --backend ollama\n"
            "\n"
            "Driving from a script or agent harness:\n"
            "  tremors query \"...\" --json -y             # one JSON doc on stdout\n"
            "  tremors query \"...\" --plan-only --json    # dry run, no FDSN\n"
            "  tremors query \"...\" --json --session-id s1   # pause -> exit 4\n"
            "  tremors resume --session-id s1 --json --approve\n"
            "\n"
            "Exit codes: 0 ok  1 failed  2 usage  3 clarification  4 awaiting\n"
            "            5 config  6 aborted\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"tremors {_tremors_version()}",
    )

    # ── Shared options (apply to all sub-commands) ─────────────────────
    shared = argparse.ArgumentParser(add_help=False)

    # --output-dir and --backend default to None rather than their real defaults
    # so that `resume` can tell "unset" from "explicitly given" and fall back to
    # the value recorded in the session. The real defaults are applied in
    # _resolve_defaults / _build_agent.
    #
    # They must NOT be set via p_resume.set_defaults(): argparse `parents=` shares
    # the *same* action objects across subparsers, and set_defaults mutates
    # action.default in place — so overriding a default on one subparser silently
    # changes it for every other one too.
    shared.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        default=None,
        metavar="DIR",
        help="Directory for output files (parquet, plots, MiniSEED). "
             f"Created if it does not exist. Default: {_DEFAULT_OUTPUT_DIR} "
             "(for `resume`, the directory the session was created with).",
    )
    shared.add_argument(
        "--backend", "-b",
        choices=list(_BACKENDS),
        default=None,
        help=f"LLM backend to use. Default: {_DEFAULT_BACKEND} (from $TREMORS_BACKEND "
             "or `tremors config` if either sets one; for `resume`, the backend the "
             "session was created with).",
    )
    shared.add_argument(
        "--model", "-m",
        default=None,
        metavar="NAME",
        help=(
            "Model name to pass to the backend. "
            f"Defaults: ollama={_DEFAULT_MODELS['ollama']!r}, "
            f"openai={_DEFAULT_MODELS['openai']!r}. "
            "The anthropic backend has no built-in default — pass --model, set "
            "ANTHROPIC_MODEL, or store one with `tremors config --save`."
        ),
    )
    shared.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        metavar="URL",
        help=(
            "Endpoint base URL. For ollama, overrides the default "
            f"{_DEFAULT_OLLAMA_URL}. For anthropic it is REQUIRED (or set "
            "ANTHROPIC_BASE_URL, or store it with `tremors config --save`) — it "
            "selects the Anthropic-compatible gateway."
        ),
    )
    shared.add_argument(
        "--no-preflight",
        dest="no_preflight",
        action="store_true",
        default=False,
        help="Skip the pre-run backend connectivity check.",
    )
    shared.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Auto-approve every human-in-the-loop gate (for scripted runs). A "
             "clarification request still stops the run — it cannot be answered "
             "automatically.",
    )
    shared.add_argument(
        "--no-interrupts",
        dest="no_interrupts",
        action="store_true",
        default=False,
        help="Disable the human-in-the-loop gates entirely. Plan verification "
             "still runs and still stops a bad plan.",
    )
    shared.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "LLM sampling temperature. If unset: ollama/openai use 0.7; "
            "anthropic omits the parameter (some gateway models reject it)."
        ),
    )
    shared.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        default=False,
        help="Write one machine-readable JSON document to stdout and route every "
             "human-facing line to stderr. For scripts and agent harnesses.",
    )
    shared.add_argument(
        "--session-id",
        dest="session_id",
        default=None,
        metavar="NAME",
        help="Run under a named, durable session (SQLite-backed) so a gate pause "
             "survives process exit and can be answered by `tremors resume`.",
    )
    shared.add_argument(
        "--session-dir",
        dest="session_dir",
        default=None,
        metavar="DIR",
        help=f"Where durable sessions live. Default: $TREMORS_SESSION_DIR or "
             f"{_DEFAULT_SESSION_DIR}",
    )
    shared.add_argument(
        "--config",
        dest="config_file",
        default=None,
        metavar="FILE",
        help="Stored settings to read (see `tremors config`). Default: "
             f"$TREMORS_CONFIG or {_DEFAULT_CONFIG_PATH}",
    )
    shared.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose/debug logging (to stderr).",
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
    p_query.add_argument(
        "--plan-only",
        dest="plan_only",
        action="store_true",
        default=False,
        help="Resolve the query to verified search parameters and stop. Sends no "
             "FDSN request and writes no files; costs one planner LLM call. "
             "Verifier findings are reported in `problems` instead of pausing.",
    )
    p_query.set_defaults(func=_cmd_query)

    # tremors resume --session-id NAME
    p_resume = sub.add_parser(
        "resume",
        parents=[shared],
        help="Answer a gate left pending by an earlier --session-id run.",
        description=(
            "Continue a durable session that stopped at a human-in-the-loop gate.\n\n"
            "The session remembers the backend, model, base URL, temperature and\n"
            "output directory of the run that created it, so no other flag is\n"
            "needed (API keys still come from the environment). Passing one of\n"
            "those flags here overrides the remembered value.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_resume.add_argument(
        "--approve",
        action="store_true",
        default=False,
        help="Approve the pending tool call(s).",
    )
    p_resume.add_argument(
        "--reject",
        default=None,
        metavar="REASON",
        help="Reject the pending call(s); the reason is sent back to the model, "
             "which re-plans.",
    )
    p_resume.add_argument(
        "--clarify",
        default=None,
        metavar="TEXT",
        help="Answer a clarification pause; the query is re-planned with this text.",
    )
    p_resume.add_argument(
        "--decision",
        default=None,
        metavar="JSON",
        help='A full decision, for edits: \'{"type":"edit","edited_action":'
             '{"name":"query_cascade","args":{"params_override":{"min_mag":6.0}}}}\'. '
             'Also accepts a list, or {"decisions": [...]}.',
    )
    p_resume.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Print what the session is waiting for and exit without answering.",
    )
    p_resume.set_defaults(func=_cmd_resume)

    # tremors sessions
    p_sessions = sub.add_parser(
        "sessions",
        parents=[shared],
        help="List or delete durable sessions.",
        description="Inspect the durable sessions available to `tremors resume`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_sessions.add_argument(
        "--delete",
        default=None,
        metavar="NAME",
        help="Delete this session's index entry.",
    )
    p_sessions.set_defaults(func=_cmd_sessions)

    # tremors config
    p_config = sub.add_parser(
        "config",
        parents=[shared],
        help="Show or store backend connection settings.",
        description=(
            "Show or store the settings TREMORS uses to reach a model backend.\n\n"
            "Settings resolve in this order:  CLI flag > environment variable >\n"
            "config file > built-in default. The file lives outside the repository\n"
            f"({_DEFAULT_CONFIG_PATH}) and is written mode 0600, so a real key or a\n"
            "private endpoint can be stored without any risk of committing it.\n\n"
            f"Connection settings ({', '.join(_BACKEND_CONFIG_KEYS)})\n"
            "are stored per backend, so a gateway model id is never handed to ollama.\n"
            "Whichever backend is in effect (--backend, else $TREMORS_BACKEND, else the\n"
            f"stored default, else {_DEFAULT_BACKEND}) names the section read or written.\n"
            f"{', '.join(_GLOBAL_CONFIG_KEYS)} are shared by all backends.\n\n"
            "Examples:\n"
            "  tremors config                          # what is in effect, and why\n"
            "  tremors config --backend ollama         # ...for a different backend\n"
            "  tremors config --json                   # same, machine-readable\n"
            "  tremors config --save --base-url https://gateway.example \\\n"
            "                 --model my-model-id --api-key sk-...\n"
            "  tremors config --save --api-key -       # read the key from stdin\n"
            "  tremors config --save --backend ollama --model gpt-oss:20b\n"
            "  tremors config --unset api_key\n\n"
            "The stored API key is never printed back — only whether one is set."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_config.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Write the settings given on this command line to the config file, "
             "merging with whatever is already stored. Connection settings land "
             "in the section for the backend in effect; --backend also records "
             "it as the new default.",
    )
    p_config.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        metavar="KEY",
        help="API key to store (only with --save). Pass '-' to read it from "
             "stdin, which keeps it out of your shell history and out of the "
             "process table.",
    )
    p_config.add_argument(
        "--unset",
        action="append",
        default=None,
        metavar="SETTING",
        help=f"Remove a stored setting. Repeatable. One of: "
             f"{', '.join(_CONFIG_KEYS)}. Connection settings "
             f"({', '.join(_BACKEND_CONFIG_KEYS)}) are removed from the section "
             f"for the backend in effect; pass --backend to pick another.",
    )
    p_config.set_defaults(func=_cmd_config)

    return parser


def main() -> None:
    global _JSON_MODE

    parser = _build_parser()
    args   = parser.parse_args()

    _JSON_MODE = bool(getattr(args, "as_json", False))

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("tremors").setLevel(logging.DEBUG)

    args.func(args)


if __name__ == "__main__":
    main()
