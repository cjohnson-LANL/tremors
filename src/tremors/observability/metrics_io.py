'''
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
'''
# metrics_io.py
from __future__ import annotations

import json
from typing import Any


def load_metrics(path: str) -> dict[str, Any]:
    """Load a telemetry metrics JSON into a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
