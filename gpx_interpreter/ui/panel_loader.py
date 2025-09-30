# gpx_interpreter/ui/panel_loader.py
from __future__ import annotations

from typing import Callable, Dict
import uuid
import streamlit as st

_REGISTRY: Dict[str, Callable] = {}


def register(name: str) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Panel '{name}' is not registered")
    return _REGISTRY[name]


# --- Per-panel namespacing helper ---
def get_ns(panel_id: str):
    """
    Return (NS, K) where NS is a stable per-session namespace for this panel
    and K(s) -> unique Streamlit key f"{NS}_{s}".
    """
    key_name = f"{panel_id}_ns"
    if key_name not in st.session_state:
        st.session_state[key_name] = f"{panel_id[:2]}_{uuid.uuid4().hex[:6]}"
    NS = st.session_state[key_name]

    def K(s: str) -> str:
        return f"{NS}_{s}"

    return NS, K
