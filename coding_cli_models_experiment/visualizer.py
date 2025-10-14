# app.py
# Streamlit app to browse instance folders, view JSONs, highlight log files, and plot time series.

import json
from pathlib import Path
from typing import List, Tuple, Optional, Union

import streamlit as st
import pandas as pd
from pygments import highlight
from pygments.lexers import TextLexer
from pygments.formatters import HtmlFormatter
import html

st.set_page_config(page_title="Instance Viewer", layout="wide")


# --------------------------
# Helpers (cached)
# --------------------------
@st.cache_data(show_spinner=False)
def list_instances(base_path: str) -> List[str]:
    p = Path(base_path).expanduser()
    if not p.exists() or not p.is_dir():
        return []
    return sorted(
        [d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )


@st.cache_data(show_spinner=False)
def list_files(instance_path: str) -> Tuple[List[str], List[str], List[str]]:
    p = Path(instance_path)
    if not p.exists():
        return [], [], []
    files = [f.name for f in p.iterdir() if f.is_file()]
    jsons = sorted([f for f in files if f.lower().endswith(".json")])
    logs = sorted(
        [f for f in files if f.lower().endswith(".log") or f.lower().endswith(".txt")]
    )
    pkls = sorted(
        [
            f
            for f in files
            if f.lower().endswith(".pkl") or f.lower().endswith(".pickle")
        ]
    )
    return jsons, logs, pkls


@st.cache_data(show_spinner=False)
def read_json_file(path: Union[str, Path]) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": f"Failed to read JSON: {e}"}


@st.cache_data(show_spinner=False)
def read_text_tail(path: Union[str, Path], max_lines: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return "".join(lines[-max_lines:])
    except Exception as e:
        return f"Failed to read log: {e}"


@st.cache_data(show_spinner=False)
def read_pickle_df(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    try:
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            df = obj
        elif isinstance(obj, pd.Series):
            df = obj.to_frame(name=obj.name or "value")
        else:
            df = pd.DataFrame(obj)

        if isinstance(df.index, pd.DatetimeIndex):
            return df.sort_index()
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    if df[col].notna().any():
                        df = df.set_index(col).sort_index()
                        break
                except Exception:
                    pass
        return df
    except Exception as e:
        return pd.DataFrame({"__error__": [f"Failed to read pickle: {e}"]})


def file_download_bytes(path: Union[str, Path]) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# --------------------------
# Log highlighting
# --------------------------
def highlight_log(content: str, wrap: bool = False) -> str:
    """
    Use Pygments to syntax-highlight log text (INFO, ERROR, WARNING, etc.)
    and return HTML suitable for Streamlit.
    """
    # Escape HTML first to avoid rendering issues
    content = html.escape(content)
    # Simple colorization rules
    content = (
        content.replace(
            "ERROR", '<span style="color:#ff4d4f;font-weight:bold;">ERROR</span>'
        )
        .replace(
            "WARNING", '<span style="color:#faad14;font-weight:bold;">WARNING</span>'
        )
        .replace("INFO", '<span style="color:#1677ff;font-weight:bold;">INFO</span>')
    )
    style = "white-space: pre-wrap;" if wrap else "white-space: pre;"
    return f"<div style='background-color:#0e1117;color:#e6e6e6;font-family:monospace;font-size:13px;padding:10px;border-radius:8px;overflow-y:auto;max-height:600px;{style}'>{content}</div>"


# --------------------------
# UI
# --------------------------
st.title("📁 Instance Browser")

with st.sidebar:
    st.header("Settings")
    base_path = st.text_input("Base path", value="", placeholder="/path/to/base_dir")
    if st.button("🔄 Refresh"):
        list_instances.clear()
        list_files.clear()

instances = []
if base_path:
    instances = list_instances(base_path)

if not base_path:
    st.info("Enter a base path in the sidebar to begin.")
    st.stop()

if not instances:
    st.warning("No subfolders found under the given base path.")
    st.stop()

instance = st.selectbox("Select instance (folder)", instances, index=0)
instance_path = str(Path(base_path).expanduser() / instance)

jsons, logs, pkls = list_files(instance_path)

# JSON selection
default_jsons = [
    f for f in ["config.json", "results.json", "metrics.json"] if f in jsons
][:2]
selected_jsons = st.multiselect(
    "Choose up to two JSON files",
    jsons,
    default=(
        default_jsons if default_jsons else (jsons[:2] if len(jsons) >= 2 else jsons)
    ),
    max_selections=2,
)

# Log + time series selection
default_log = "output.log" if "output.log" in logs else (logs[-1] if logs else None)
selected_log = st.selectbox(
    "Log file",
    ["— none —"] + logs,
    index=(0 if not default_log else (logs.index(default_log) + 1)),
)
tail_lines = st.slider(
    "Show last N lines", min_value=50, max_value=5000, value=400, step=50
)
default_pkl = (
    "timeseries.pkl" if "timeseries.pkl" in pkls else (pkls[0] if pkls else None)
)
selected_pkl = st.selectbox(
    "Time series (.pkl)",
    ["— none —"] + pkls,
    index=(0 if not default_pkl else (pkls.index(default_pkl) + 1)),
)

st.divider()

# --------------------------
# JSON display
# --------------------------
if selected_jsons:
    cols = st.columns(len(selected_jsons))
    for i, jf in enumerate(selected_jsons):
        with cols[i]:
            st.subheader(f"🧾 {jf}")
            jpath = Path(instance_path) / jf
            data = read_json_file(jpath)
            if isinstance(data, dict) and "__error__" in data:
                st.error(data["__error__"])
            else:
                st.json(data, expanded=False)
            st.download_button(
                "Download JSON",
                data=file_download_bytes(jpath),
                file_name=jf,
                mime="application/json",
                use_container_width=True,
            )
else:
    st.info("Select up to two JSON files to display.")

# --------------------------
# Highlighted log display
# --------------------------
st.divider()
st.subheader("📝 Log")

if selected_log != "— none —":
    lpath = Path(instance_path) / selected_log
    content = read_text_tail(lpath, tail_lines) or "(empty)"

    # col1, col2 = st.columns([1, 1])
    # with col1:
    #     wrap = st.toggle("Wrap long lines", value=False, help="Toggle soft wrap for very long lines.")
    # with col2:
    #     height = st.slider("Viewer height (px)", 200, 1200, 450, 50)

    highlighted = highlight_log(content)
    st.markdown(
        f"<div style=overflow-y:auto'>{highlighted}</div>", unsafe_allow_html=True
    )

    st.download_button(
        "Download log",
        data=file_download_bytes(lpath),
        file_name=selected_log,
        mime="text/plain",
        use_container_width=True,
    )
else:
    st.caption("No log selected.")

st.divider()
st.subheader("📈 Time Series")

if selected_pkl != "— none —":
    ppath = Path(instance_path) / selected_pkl

    # Read strictly as DataFrame
    try:
        obj = pd.read_pickle(ppath)
        df = pd.DataFrame(obj)
    except Exception as e:
        df = None
        st.error(f"Failed to read pickle: {e}")

    st.line_chart(df)

    # Optional: export what you plotted
    st.download_button(
        "Download plotted data (CSV)",
        data=df.to_csv(index=True).encode("utf-8"),
        file_name=Path(selected_pkl).with_suffix(".csv").name,
        mime="text/csv",
        use_container_width=True,
    )


# --------------------------
# Footer
# --------------------------
st.divider()
st.caption(f"Base: {Path(base_path).expanduser()} — Instance: {instance}")
