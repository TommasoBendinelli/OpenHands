#!/usr/bin/env python3
"""Generate an interactive HTML dashboard for a single evaluation run."""

from __future__ import annotations

import argparse
import base64
import functools
import html
import http.server
import json
import os
import re
import socketserver
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TIMESTAMP_PATTERN = re.compile(r"^\[\d{4}-\d{2}-\d{2}T")


def _read_text_block(lines: List[str], start_idx: int) -> Tuple[str, int]:
    """Return the text block starting at ``start_idx`` until the next timestamp."""

    collected: List[str] = []
    idx = start_idx
    while idx < len(lines):
        line = lines[idx]
        if TIMESTAMP_PATTERN.match(line):
            break
        collected.append(line)
        idx += 1
    return "\n".join(collected).strip(), idx


def extract_sections(log_path: Path) -> Dict[str, str]:
    """Parse a Codex log to find user instructions and the final answer."""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    sections: Dict[str, str] = {}

    # User instructions
    for idx, line in enumerate(lines):
        if "User instructions:" in line:
            block, _ = _read_text_block(lines, idx + 1)
            sections["user_instructions"] = block
            break

    # Model output: last "Final answer" style block in the log
    final_idx = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith("Final answer:"):
            final_idx = idx

    if final_idx != -1:
        block, _ = _read_text_block(lines, final_idx)
        sections["final_answer"] = block

    # Any trailing model-produced summary (after the last "thinking" block)
    completion_idx = -1
    for idx, line in enumerate(reversed(lines)):
        if TIMESTAMP_PATTERN.match(line):
            completion_idx = len(lines) - idx
            break

    if completion_idx not in (-1, len(lines)):
        block = "\n".join(lines[completion_idx:]).strip()
        if block:
            sections["tail"] = block

    return sections


def gather_images(asset_dirs: Sequence[Path]) -> Dict[str, List[Tuple[str, str]]]:
    """Collect PNG/JPG images from each asset directory.

    Parameters
    ----------
    asset_dirs:
        Directories to search for images.

    Returns
    -------
    dict
        Mapping of display category to (label, data-url) tuples.
    """

    categories: Dict[str, List[Tuple[str, str]]] = {
        "Model input": [],
        "Model output": [],
    }

    seen: set[Path] = set()
    patterns = ("*.png", "*.jpg", "*.jpeg")

    for asset_dir in asset_dirs:
        if not asset_dir.exists() or not asset_dir.is_dir():
            continue

        for pattern in patterns:
            for path in sorted(asset_dir.rglob(pattern)):
                if path.is_dir():
                    continue

                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)

                try:
                    rel = path.relative_to(asset_dir).as_posix()
                except ValueError:
                    rel = path.name

                display_label = f"{asset_dir.name}/{rel}" if rel else asset_dir.name
                category = "Model input" if "diagrams" in path.parts else "Model output"
                data_url = _image_to_data_url(path)
                categories[category].append((display_label, data_url))

    return {k: v for k, v in categories.items() if v}


def _image_to_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() in {".png"} else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_html(
    source: Path,
    log_path: Path,
    sections: Dict[str, str],
    images: Dict[str, List[Tuple[str, str]]],
    output_path: Path,
    asset_dirs: Sequence[Path],
) -> str:
    title_label = log_path.parent.name or log_path.name
    def render_text(label: str, content: str) -> str:
        escaped = html.escape(content).replace("\n", "<br>") if content else "<em>Not available.</em>"
        return f"""
        <section class="card">
            <h2>{html.escape(label)}</h2>
            <div class="card-body">{escaped}</div>
        </section>
        """

    image_control = "<p><em>No images found for this run.</em></p>"
    if images:
        options_html: List[str] = []
        for category, files in images.items():
            opts = "\n".join(
                f"<option value=\"{html.escape(data)}\">{html.escape(rel_path)}" for rel_path, data in files
            )
            options_html.append(f"<optgroup label=\"{html.escape(category)}\">{opts}</optgroup>")

        first_image = next(iter(images.values()))[0][1]
        image_control = f"""
        <div class="card">
            <h2>Visuals</h2>
            <div class="card-body">
                <label for="image-select">Choose an image:</label>
                <select id="image-select">
                    {''.join(options_html)}
                </select>
                <div class="image-container">
                    <img id="preview" src="{first_image}" alt="Selected visual" />
                </div>
            </div>
        </div>
        """

    template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Run Visualizer - {html.escape(title_label)}</title>
    <style>
        :root {{
            color-scheme: light dark;
            font-family: system-ui, sans-serif;
            background: #f6f7fb;
            color: #1f2933;
        }}
        body {{
            margin: 0;
            padding: 2rem;
        }}
        h1 {{
            margin-top: 0;
            font-size: 1.8rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 12px 24px -14px rgba(31, 41, 51, 0.6);
            padding: 1.25rem;
        }}
        .card-body {{
            margin-top: 0.75rem;
            line-height: 1.5;
            white-space: normal;
            font-size: 0.95rem;
        }}
        select {{
            width: 100%;
            padding: 0.5rem;
            margin-top: 0.5rem;
            border-radius: 8px;
            border: 1px solid #d0d7de;
            background: #fff;
            font-size: 0.95rem;
        }}
        .image-container {{
            margin-top: 1rem;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 8px 18px -12px rgba(15, 23, 42, 0.8);
        }}
        footer {{
            margin-top: 2rem;
            font-size: 0.85rem;
            color: #52606d;
        }}
    </style>
</head>
<body>
    <h1>Run Visualizer</h1>
    <p><strong>Source:</strong> {html.escape(str(source))}</p>
    <p><strong>Log file:</strong> {html.escape(str(log_path))}</p>
    <p><strong>Visualizer file:</strong> {html.escape(str(output_path))}</p>
    <p><strong>Image search roots:</strong> {html.escape(', '.join(str(d) for d in asset_dirs) or 'None')}</p>
    <div class="grid">
        {render_text('Model input', sections.get('user_instructions', ''))}
        {render_text('Model output', sections.get('final_answer', ''))}
        {image_control}
    </div>
    <footer>Generated by visualize_run.py</footer>
    <script>
    const select = document.getElementById('image-select');
    if (select) {{
        const img = document.getElementById('preview');
        select.addEventListener('change', () => {{
            img.src = select.value;
        }});
    }}
    </script>
</body>
</html>
"""

    return template


def _serve_directory(root: Path, output_path: Path, host: str, port: int) -> None:
    handler_cls = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with socketserver.ThreadingTCPServer((host, port), handler_cls) as httpd:
        bind_host, bind_port = httpd.server_address
        display_host = host if host not in {"0.0.0.0", ""} else bind_host
        url = f"http://{display_host}:{bind_port}/{output_path.name}"
        print(json.dumps({"generated": str(output_path), "url": url}))
        print(f"Serving {root} at {url} (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Codex evaluation runs.")
    parser.add_argument(
        "source",
        type=Path,
        help="Path to either an evaluation directory or a standalone output.log file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the generated HTML. Defaults to run_dir / 'run_visualizer.html'",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a simple HTTP server rooted at the run directory after generating the HTML.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/IP to bind the HTTP server (use 0.0.0.0 to listen on all interfaces).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port for the HTTP server (0 lets the OS pick an available port).",
    )
    parser.add_argument(
        "--assets",
        action="append",
        type=Path,
        help="Additional directories to search for images (can be repeated).",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source path not found: {source}")

    def create_run_page(log_path: Path, extra_assets: Sequence[Path], explicit_output: Path | None = None):
        run_base = log_path.parent
        asset_dirs = [run_base]
        asset_dirs.extend(extra_assets)
        output_path = explicit_output or (run_base / "run_visualizer.html")
        sections = extract_sections(log_path)
        images = gather_images(asset_dirs)
        html_text = build_html(log_path, log_path, sections, images, output_path, asset_dirs)
        output_path.write_text(html_text, encoding="utf-8")
        return output_path, sections

    def build_multi_index(entries: List[Dict[str, str]], output_path: Path) -> None:
        grouped: Dict[str, List[Dict[str, str]]] = {}
        for entry in entries:
            day = entry.get('day', 'Unknown')
            grouped.setdefault(day, []).append(entry)

        sorted_days = sorted(grouped.keys(), reverse=True)
        sections: List[str] = []
        for day in sorted_days:
            rows_html = "".join(
                f"<tr><td>{html.escape(item['title'])}</td><td><code>{html.escape(item['log'])}</code></td><td><a href='{html.escape(item['html'])}' target='_blank'>Open</a></td><td>{html.escape(item.get('final', 'N/A'))}</td></tr>"
                for item in sorted(grouped[day], key=lambda x: (x.get('title', ''), x.get('log', '')))
            )
            sections.append(
                f"""
                <section>
                    <h2>{html.escape(day)}</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Run</th>
                                <th>Log</th>
                                <th>Visualizer</th>
                                <th>Final answer</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </section>
                """
            )

        template = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>Evaluation Visualizer Index</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 2rem; background: #f6f7fb; color: #1f2933; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #d0d7de; }}
        th {{ background: #e4e9f2; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.75rem; }}
        tr:hover {{ background: rgba(79, 70, 229, 0.08); }}
        a {{ color: #4338ca; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        h1 {{ margin-bottom: 0.5rem; }}
        section {{ margin-top: 2.5rem; }}
    </style>
</head>
<body>
    <h1>Evaluation Visualizer Index</h1>
    <p>Generated {html.escape(str(output_path))}</p>
    {''.join(sections)}
</body>
</html>
"""

        output_path.write_text(template, encoding="utf-8")

    extra_assets = [path.expanduser().resolve() for path in (args.assets or [])]

    if source.is_dir():
        log_path = source / "output.log"
        if log_path.exists():
            run_html, sections = create_run_page(log_path, extra_assets, explicit_output=args.output)
            if args.serve:
                _serve_directory(run_html.parent, run_html, args.host, args.port)
            else:
                print(json.dumps({"generated": str(run_html), "url": None}))
            return

        candidates = sorted(source.rglob("output.log"))
        if not candidates:
            raise SystemExit(f"No output.log files found under {source}")

        output_path = (args.output or (source / "evaluation_visualizer.html")).expanduser().resolve()
        entries: List[Dict[str, str]] = []
        for log_path in candidates:
            run_html, sections = create_run_page(log_path, extra_assets)
            try:
                rel_html_path = run_html.relative_to(output_path.parent)
                rel_html = rel_html_path.as_posix()
            except ValueError:
                rel_html = os.path.relpath(run_html, output_path.parent)
            final_line = sections.get("final_answer", "").splitlines()
            final_text = final_line[0] if final_line else ""
            try:
                rel_parts = log_path.relative_to(source).parts
                day = rel_parts[0] if rel_parts else log_path.parent.name
            except ValueError:
                day = log_path.parent.parent.name if log_path.parent.parent else log_path.parent.name
            entries.append(
                {
                    "title": log_path.parent.name,
                    "log": str(log_path),
                    "html": rel_html,
                    "final": final_text,
                    "day": day,
                }
            )

        build_multi_index(entries, output_path)

        if args.serve:
            _serve_directory(output_path.parent, output_path, args.host, args.port)
        else:
            print(json.dumps({"generated": str(output_path), "url": None}))
        return

    # Source is a file (single log)
    output_path = (args.output or (source.parent / "run_visualizer.html")).expanduser().resolve()
    run_html, _ = create_run_page(source, extra_assets, explicit_output=output_path)

    if args.serve:
        _serve_directory(run_html.parent, run_html, args.host, args.port)
    else:
        print(json.dumps({"generated": str(run_html), "url": None}))


if __name__ == "__main__":
    main()
