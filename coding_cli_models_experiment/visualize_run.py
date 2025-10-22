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
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


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


def extract_sections(log_path: Path) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """Parse an evaluation log (Codex or Claude) into display sections and events."""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip()

    if stripped.startswith("{"):
        try:
            sections, events = _extract_claude_log(log_path, [line for line in stripped.splitlines() if line])
        except json.JSONDecodeError:
            sections, events = _extract_codex_log(text.splitlines())
    else:
        sections, events = _extract_codex_log(text.splitlines())

    # Always prioritize results.json for final answer (more reliable)
    answer = _load_results_answer(log_path)
    if answer:
        sections['final_answer'] = answer

    # Load correct answer from config.yaml
    correct_answer = _load_correct_answer_from_config(log_path)
    if correct_answer:
        sections['correct_answer'] = correct_answer

    if not sections.get('user_instructions'):
        instructions = _load_prompt_from_config(log_path)
        if instructions:
            sections['user_instructions'] = instructions

    return sections, events


def _extract_codex_log(lines: List[str]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    sections: Dict[str, str] = {}

    for idx, line in enumerate(lines):
        if "User instructions:" in line:
            block, _ = _read_text_block(lines, idx + 1)
            sections["user_instructions"] = block
            break

    final_idx = -1
    for idx, line in enumerate(lines):
        if line.strip().startswith("Final answer:"):
            final_idx = idx

    if final_idx != -1:
        block, _ = _read_text_block(lines, final_idx)
        sections["final_answer"] = block

    completion_idx = -1
    for idx, line in enumerate(reversed(lines)):
        if TIMESTAMP_PATTERN.match(line):
            completion_idx = len(lines) - idx
            break

    if completion_idx not in (-1, len(lines)):
        block = "\n".join(lines[completion_idx:]).strip()
        if block:
            sections["tail"] = block

    events: List[Dict[str, str]] = []
    pending_cmd: Dict[str, str] | None = None

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = TIMESTAMP_PATTERN.match(line)
        if not match:
            idx += 1
            continue

        end_bracket = line.find(']')
        timestamp = line[1:end_bracket]
        rest = line[end_bracket + 1 :].strip()

        if rest.startswith('exec '):
            cwd = None
            if ' in ' in rest:
                cmd_part, _, cwd = rest.partition(' in ')
                command_text = cmd_part
            else:
                command_text = rest
            pending_cmd = {
                'type': 'command',
                'timestamp': timestamp,
                'command': command_text,
            }
            if cwd:
                pending_cmd['cwd'] = cwd
            events.append(pending_cmd)

        elif rest.startswith('bash ') or rest.startswith('python '):
            if pending_cmd is not None:
                pending_cmd['result'] = rest
                output_lines: List[str] = []
                j = idx + 1
                while j < len(lines) and not TIMESTAMP_PATTERN.match(lines[j]):
                    output_lines.append(lines[j])
                    j += 1
                pending_cmd['output'] = '\n'.join(output_lines).strip()
                pending_cmd = None
                idx = j - 1

        elif rest.startswith('thinking'):
            block, next_idx = _read_text_block(lines, idx + 1)
            events.append(
                {
                    'type': 'thinking',
                    'timestamp': timestamp,
                    'text': block,
                }
            )
            idx = next_idx - 1

        elif rest.startswith('tokens used'):
            events.append(
                {
                    'type': 'info',
                    'timestamp': timestamp,
                    'text': rest,
                }
            )

        idx += 1

    return sections, events


def _extract_claude_log(log_path: Path, lines: Sequence[str]) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    sections: Dict[str, str] = {}
    events: List[Dict[str, str]] = []
    command_events: Dict[str, Dict[str, str]] = {}
    final_answer_candidate: str | None = None

    for index, raw_line in enumerate(lines, start=1):
        obj = json.loads(raw_line)
        timestamp = _format_claude_timestamp(index, obj)
        entry_type = obj.get('type')

        if entry_type == 'assistant':
            message = obj.get('message', {})
            contents = message.get('content') or []
            for item in contents:
                item_type = item.get('type')
                if item_type == 'text':
                    text = (item.get('text') or '').strip()
                    if not text:
                        continue
                    events.append({'type': 'thinking', 'timestamp': timestamp, 'text': text})
                    if 'final answer' in text.lower():
                        final_answer_candidate = text
                elif item_type == 'tool_use':
                    tool_id = item.get('id')
                    tool_name = (item.get('name') or '').strip()
                    tool_input = item.get('input')
                    command_text, description = _summarize_tool_use(tool_name, tool_input)
                    if tool_name.lower() == 'bash' and command_text:
                        event: Dict[str, str] = {
                            'type': 'command',
                            'timestamp': timestamp,
                            'command': command_text,
                        }
                        status_parts: List[str] = []
                        if description:
                            status_parts.append(description)
                        if status_parts:
                            event['result'] = '\n'.join(status_parts)
                        events.append(event)
                        if tool_id:
                            command_events[tool_id] = event
                    elif command_text:
                        info_body = description or command_text
                        if tool_name:
                            info_text = f"{tool_name}: {info_body}"
                        else:
                            info_text = info_body
                        events.append({'type': 'info', 'timestamp': timestamp, 'text': info_text})

        elif entry_type == 'user':
            message = obj.get('message', {})
            contents = message.get('content') or []
            for item in contents:
                if item.get('type') != 'tool_result':
                    continue
                tool_id = item.get('tool_use_id')
                event = command_events.get(tool_id)
                output_text = _stringify_tool_result(item.get('content'))
                status = 'Status: error' if item.get('is_error') else 'Status: success'
                if event:
                    existing_result = event.get('result')
                    if existing_result:
                        event['result'] = f"{existing_result}\n{status}"
                    else:
                        event['result'] = status
                    if output_text:
                        event['output'] = output_text
                elif output_text:
                    events.append({'type': 'info', 'timestamp': timestamp, 'text': output_text})

    if final_answer_candidate:
        sections['final_answer'] = final_answer_candidate

    return sections, events


def _format_claude_timestamp(index: int, obj: Dict[str, Any]) -> str:
    message = obj.get('message') or {}
    message_id = message.get('id')
    if isinstance(message_id, str) and message_id:
        suffix = message_id.split('_')[-1][:8]
        return f"{index:03d}-{suffix}"
    uuid = obj.get('uuid')
    if isinstance(uuid, str) and uuid:
        return f"{index:03d}-{uuid[:8]}"
    return f"{index:03d}"


def _summarize_tool_use(tool_name: str, tool_input: Any) -> Tuple[str, str]:
    if isinstance(tool_input, dict):
        command = tool_input.get('command')
        description = tool_input.get('description') or ''
        if isinstance(command, str) and command.strip():
            return command.strip(), description.strip()
        if description:
            return description.strip(), ''
        return json.dumps(tool_input, ensure_ascii=False), ''
    if isinstance(tool_input, str):
        return tool_input.strip(), ''
    if tool_input is None:
        return tool_name or '', ''
    return json.dumps(tool_input, ensure_ascii=False), ''


def _stringify_tool_result(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        lines: List[str] = []
        for entry in content:
            if isinstance(entry, str):
                lines.append(entry)
            elif isinstance(entry, dict):
                # Check if this is an image entry
                if entry.get('type') == 'image':
                    # Return a special marker that we'll handle in rendering
                    source = entry.get('source', {})
                    if source.get('type') == 'base64':
                        media_type = source.get('media_type', 'image/png')
                        data = source.get('data', '')
                        lines.append(f"__IMAGE__{media_type}|{data}__IMAGE_END__")
                    continue
                lines.append(json.dumps(entry, ensure_ascii=False))
            else:
                lines.append(json.dumps(entry, ensure_ascii=False))
        return '\n'.join(line.strip() for line in lines if line.strip())
    if content is None:
        return ''
    return json.dumps(content, ensure_ascii=False)


def _load_results_answer(log_path: Path) -> str | None:
    """Load the final answer from results.json file."""
    results_path = log_path.with_name('results.json')
    if not results_path.exists():
        return None
    try:
        data = json.loads(results_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        # Try multiple possible keys for the answer
        for key in ('final_answer', 'answer', 'result'):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _load_prompt_from_config(log_path: Path) -> str | None:
    if yaml is None:
        return None
    config_path = log_path.with_name('config.yaml')
    if not config_path.exists():
        return None
    try:
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    except Exception:  # pragma: no cover - best-effort extraction
        return None
    prompt = config.get('prompt') if isinstance(config, dict) else None
    if isinstance(prompt, str):
        return prompt.strip()
    return None


def _load_correct_answer_from_config(log_path: Path) -> str | None:
    """Load the correct answer from config.yaml file."""
    if yaml is None:
        return None
    config_path = log_path.with_name('config.yaml')
    if not config_path.exists():
        return None
    try:
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    except Exception:  # pragma: no cover - best-effort extraction
        return None
    correct_answer = config.get('correct_answer') if isinstance(config, dict) else None
    if isinstance(correct_answer, str):
        return correct_answer.strip()
    return None


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
        "Model response": [],
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
                category = "Model input" if "diagrams" in path.parts else "Model response"
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
    events: List[Dict[str, str]],
    images: Dict[str, List[Tuple[str, str]]],
    output_path: Path,
    asset_dirs: Sequence[Path],
) -> str:
    title_label = log_path.parent.name or log_path.name

    def render_text(label: str, content: str, collapsible: bool = False, collapsed_by_default: bool = False, is_correct_answer: bool = False) -> str:
        escaped = html.escape(content).replace("\n", "<br>") if content else "<em>Not available.</em>"
        card_class = "card correct-answer" if is_correct_answer else "card"
        if collapsible:
            open_attr = "" if collapsed_by_default else " open"
            return f"""
        <article class="{card_class}">
            <header class="card-header"><h2>{html.escape(label)}</h2></header>
            <div class="card-body">
                <details class="content-details"{open_attr}>
                    <summary class="content-summary">Click to expand/collapse</summary>
                    <div class="content-body">{escaped}</div>
                </details>
            </div>
        </article>
        """
        return f"""
        <article class="{card_class}">
            <header class="card-header"><h2>{html.escape(label)}</h2></header>
            <div class="card-body">{escaped}</div>
        </article>
        """

    metadata_pairs = [
        ("Source", str(source)),
        ("Log file", str(log_path)),
        ("Visualizer file", str(output_path)),
        ("Image search roots", ", ".join(str(d) for d in asset_dirs) or "None"),
    ]
    metadata_html = "".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd>" for label, value in metadata_pairs
    )

    image_control = """
        <article class="card">
            <header class="card-header"><h2>Visuals</h2></header>
            <div class="card-body"><em>No images found for this run.</em></div>
        </article>
    """
    if images:
        options_html: List[str] = []
        first_label = ""
        first_image = ""
        multi_category = len(images) > 1
        first_found = False
        for category, files in images.items():
            option_tags: List[str] = []
            for rel_path, data in files:
                option_label = rel_path if not multi_category else f"{rel_path} ({category})"
                option_tags.append(
                    f'<option value="{html.escape(data)}">{html.escape(option_label)}</option>'
                )
                if not first_found:
                    first_label = option_label
                    first_image = data
                    first_found = True
            if multi_category:
                options_html.append(
                    f'<optgroup label="{html.escape(category)}">{"".join(option_tags)}</optgroup>'
                )
            else:
                options_html.extend(option_tags)

        image_control = f"""
        <article class="card visuals-card">
            <header class="card-header"><h2>Visuals</h2></header>
            <div class="card-body">
                <label class="vis-label" for="image-select">Choose an image</label>
                <select id="image-select" aria-label="Select a captured visual">
                    {''.join(options_html)}
                </select>
                <figure class="image-container">
                    <img id="preview" src="{first_image}" alt="Selected visual" />
                    <figcaption id="image-caption">{html.escape(first_label)}</figcaption>
                </figure>
            </div>
        </article>
        """


    def render_output_content(output_text: str) -> str:
        """Render output text, handling special cases like images and code."""
        if not output_text:
            return ""

        # Check for image markers
        image_pattern = re.compile(r'__IMAGE__([^|]+)\|([^_]+)')
        match = image_pattern.search(output_text)

        if match:
            media_type = match.group(1)
            base64_data = match.group(2)
            # Remove the image marker from text and add image display
            text_before = output_text[:match.start()]
            text_after = output_text[match.end():]

            html_parts = []
            if text_before.strip():
                html_parts.append(f"<pre class='event-block'>{html.escape(text_before.strip())}</pre>")

            html_parts.append(f"""<div class='image-output'>
                <img src='data:{html.escape(media_type)};base64,{html.escape(base64_data)}'
                     alt='Output image'
                     style='max-width: 100%; border-radius: 8px; margin: 0.5rem 0;' />
            </div>""")

            text_after = text_after.strip("__IMAGE_END__")
            if text_after.strip():
                html_parts.append(f"<pre class='event-block'>{html.escape(text_after.strip())}</pre>")

            return ''.join(html_parts)

        # Handle Python code formatting - detect if it looks like inline Python code
        if output_text.strip().startswith('python3 <<') or 'import ' in output_text[:100]:
            # Try to format Python code with proper line breaks
            # Replace common patterns that should have newlines
            formatted = output_text
            formatted = re.sub(r'\s+import\s+', '\nimport ', formatted)
            formatted = re.sub(r'\s+from\s+', '\nfrom ', formatted)
            formatted = re.sub(r'\s+def\s+', '\ndef ', formatted)
            formatted = re.sub(r'\s+class\s+', '\nclass ', formatted)
            formatted = re.sub(r'\s+if\s+__name__', '\nif __name__', formatted)
            formatted = re.sub(r'#\s*', '\n# ', formatted)
            # Clean up excessive newlines
            formatted = re.sub(r'\n\n+', '\n\n', formatted)
            return f"<pre class='event-block'>{html.escape(formatted.strip())}</pre>"

        return ""

        # return f"<pre class='event-block'>{html.escape(output_text)}</pre>"

    def extract_and_prettyfy_python(source: str, *, line_length: int = 88) -> str:
        """
        Extract Python code from a bash heredoc (if present), unescape HTML entities,
        and pretty-format it. Prefers 'black', then 'autopep8', then a minimal AST round-trip.

        Parameters
        ----------
        source : str
            Input text that may contain a heredoc like:
                python3 << 'EOF'
                ... python code ...
                EOF
            or just raw Python code.
        line_length : int
            Target line length for formatters that support it.

        Returns
        -------
        str
            A formatted Python code string.
        """
        # 1) Unescape HTML entities early so regex can see real quotes/angles
        text = html.unescape(source)

        # 2) Try to extract a heredoc payload if it exists
        #    Supports delimiters like EOF, "EOF", 'EOF', and allows leading/trailing spaces.
        heredoc_pattern = re.compile(
            r"""
            ^\s*python(?:3)?\s*<<\s*(?P<q>['"]?)(?P<delim>[A-Za-z0-9_]+)(?P=q)\s*[\r\n]+   # opener
            (?P<code>.*?)                                                                  # code
            ^\s*(?P=delim)\s*$                                                             # closer
            """,
            re.DOTALL | re.MULTILINE | re.VERBOSE,
        )

        m = heredoc_pattern.search(text)
        code = m.group("code") if m else text  # if no heredoc, assume the whole input is python

        # Strip BOMs / leading/trailing whitespace noise
        code = code.lstrip("\ufeff").strip("\n\r ")

        # 3) Try best-in-class formatters, gracefully degrading

        # Try black
        try:
            import black  # type: ignore
            mode = black.Mode(line_length=line_length)
            return black.format_str(code, mode=mode)
        except Exception:
            pass

        # Try autopep8
        try:
            import autopep8  # type: ignore
            return autopep8.fix_code(code, options={"max_line_length": line_length})
        except Exception:
            pass

        # Try minimal AST round-trip (Python 3.9+ has ast.unparse)
        try:
            import ast
            if hasattr(ast, "unparse"):
                tree = ast.parse(code)
                # ast.unparse returns syntactically valid code, but not perfectly styled.
                unparsed = ast.unparse(tree)
                # Make it a bit nicer w/ a simple newline policy.
                return (unparsed.rstrip() + "\n")
        except Exception:
            pass

        # If everything fails, return the cleaned code unchanged.
        return code


    def _escape_code(s: str) -> str:
        return html.escape(s, quote=False)

    def _code_block(code: str, lang: str = "bash") -> str:
        return f"<pre class='event-block'><code class='language-{lang}'>{_escape_code(code)}</code></pre>"

    def _is_python_command(cmd: str) -> bool:
        # be forgiving about whitespace and variants like `python`, `python3`, `python - <<EOF`, etc.
        c = cmd.lstrip()
        return c.startswith("python") and ("<<" in c or " -c " in c or c.strip() == "python" or c.strip().startswith("python3"))

    def render_event(event: Dict[str, str]) -> str:
        etype = event.get("type")
        timestamp = html.escape(event.get("timestamp", ""))
        if etype == "command":
            raw_command = event.get("command", "") or ""
            command_text = raw_command  # keep raw for detection/blocks (avoid early escaping)
            is_py = _is_python_command(command_text)

            # Decide how to show the summary line
            if is_py:
                # Show a concise summary label instead of dumping the whole heredoc in the <summary>
                summary_cmd_html = "<span class='event-label code-kind'>Python</span>"
            else:
                summary_cmd_html = f"<code class='event-command language-bash'>{_escape_code(command_text)}</code>"

            # Build the command/body sections
            if is_py:
                # Prettify and render Python
                try:
                    pretty_py = extract_and_prettyfy_python(command_text)
                except Exception:
                    pretty_py = command_text  # graceful fallback

                command_section_html = (
                    # + "<h4>Python</h4>"
                    _code_block(pretty_py, "python")
                )
            else:
                command_section_html = "<h4>Command</h4>" + _code_block(command_text, "bash")

            cwd_html = (
                f"<div class='event-meta'><strong>Working dir:</strong> {html.escape(event['cwd'])}</div>"
                if event.get("cwd")
                else ""
            )
            result_html = (
                f"<h4>Result</h4>{_code_block(event.get('result', ''), 'text')}"
                if event.get("result")
                else ""
            )
            output_html = (
                f"<h4>Output</h4>{render_output_content(event.get('output', ''))}"
                if event.get("output")
                else ""
            )

            return f"""
                <details class="event">
                    <summary>
                        <span class="event-time">{timestamp}</span>
                        {summary_cmd_html}
                    </summary>
                    <div class="event-body">
                        {cwd_html}
                        {command_section_html}
                        {result_html}
                        {output_html}
                    </div>
                </details>
            """

        if etype == "thinking":
            text_html = html.escape(event.get("text", ""))
            return f"""
                <details class="event">
                    <summary>
                        <span class="event-time">{timestamp}</span>
                        <span class="event-label">Thinking</span>
                    </summary>
                    <div class="event-body">
                        <pre class='event-block'><code class="language-text">{text_html}</code></pre>
                    </div>
                </details>
            """

        if etype == "info":
            text_html = render_output_content(html.escape(event.get("text", "")))
            return f"""
                <div class="event info-event">
                    <span class="event-time">{timestamp}</span>
                    <span class="event-label">{text_html}</span>
                </div>
            """
        return ""

    if events:
        timeline_html = "".join(render_event(event) for event in events)
        event_block = f"""
        <article class="card">
            <header class="card-header"><h2>Model activity</h2></header>
            <div class="card-body event-timeline">
                {timeline_html}
            </div>
        </article>
        """
    else:
        event_block = """
        <article class="card">
            <header class="card-header"><h2>Model activity</h2></header>
            <div class="card-body"><em>No model activity recorded.</em></div>
        </article>
        """

    tail_block = render_text("Completion tail", sections["tail"]) if sections.get("tail") else ""

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
            --bg: #f6f7fb;
            --card: #ffffff;
            --text: #1f2933;
            --muted: #64748b;
            background: var(--bg);
            color: var(--text);
        }}
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }}
        header.page-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            max-width: 960px;
            margin: 0 auto;
            padding: 2rem 1.5rem 1rem;
        }}
        header.page-header h1 {{
            margin: 0;
            font-size: clamp(1.6rem, 2vw + 1rem, 2.1rem);
        }}
        .content-details {{
            margin-top: 0.5rem;
        }}
        .content-summary {{
            cursor: pointer;
            font-weight: 600;
            color: var(--muted);
            font-size: 0.9rem;
            padding: 0.5rem 0;
            list-style: none;
        }}
        .content-summary::-webkit-details-marker {{
            display: none;
        }}
        .content-summary::before {{
            content: '▶ ';
            display: inline-block;
            transition: transform 0.2s ease;
        }}
        .content-details[open] .content-summary::before {{
            transform: rotate(90deg);
        }}
        .content-body {{
            margin-top: 0.75rem;
        }}
        button.meta-toggle {{
            border: none;
            background: #4338ca;
            color: #f8fafc;
            padding: 0.65rem 1.2rem;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.15s ease, background 0.2s ease;
        }}
        button.meta-toggle:hover {{
            transform: translateY(-1px);
            background: #4f46e5;
        }}
        button.meta-toggle:focus {{
            outline: 3px solid rgba(79, 70, 229, 0.4);
            outline-offset: 2px;
        }}
        main {{
            max-width: 960px;
            margin: 0 auto;
            padding: 0 1.5rem 3rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }}
        .card {{
            background: var(--card);
            border-radius: 16px;
            box-shadow: 0 18px 40px -32px rgba(15, 23, 42, 0.9);
            overflow: hidden;
        }}
        .card.correct-answer {{
            border-left: 4px solid #10b981;
        }}
        .card.correct-answer .card-header h2 {{
            color: #059669;
        }}
        .card-header {{
            padding: 1.2rem 1.5rem 0;
        }}
        .card-header h2 {{
            margin: 0;
            font-size: 1.2rem;
        }}
        .card-body {{
            padding: 1rem 1.5rem 1.5rem;
            line-height: 1.6;
            font-size: 0.95rem;
        }}
        .card-body em {{
            color: var(--muted);
        }}
        .visuals-card select {{
            width: 100%;
            padding: 0.6rem 0.9rem;
            border-radius: 10px;
            border: 1px solid #d0d7de;
            background: #fff;
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }}
        .vis-label {{
            display: block;
            font-weight: 600;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }}
        .image-container {{
            margin-top: 1rem;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 14px;
            box-shadow: 0 20px 45px -35px rgba(15, 23, 42, 0.8);
        }}
        .image-container figcaption {{
            margin-top: 0.6rem;
            font-size: 0.85rem;
            color: var(--muted);
        }}
        .event-timeline {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}
        .event summary {{
            cursor: pointer;
            display: grid;
            grid-template-columns: minmax(0, 9rem) 1fr;
            gap: 1rem;
            align-items: baseline;
            font-weight: 600;
        }}
        .event-time {{
            font-size: 0.82rem;
            color: var(--muted);
        }}
        .event-command {{
            font-family: ui-monospace, SFMono-Regular, SFMono, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.9rem;
            overflow-wrap: anywhere;
        }}
        .event-body {{
            margin-top: 0.9rem;
        }}
        .event-block {{
            background: #0f172a;
            color: #e2e8f0;
            border-radius: 10px;
            padding: 0.8rem;
            overflow-x: auto;
            font-size: 0.85rem;
            line-height: 1.55;
        }}
        .event-label {{
            font-weight: 500;
        }}
        .event-meta {{
            font-size: 0.85rem;
            color: var(--muted);
            margin-bottom: 0.75rem;
        }}
        .info-event {{
            display: flex;
            gap: 0.75rem;
            align-items: baseline;
            font-size: 0.85rem;
            color: var(--muted);
        }}
        h4 {{
            font-size: 0.85rem;
            margin: 1rem 0 0.35rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
        }}
        footer {{
            max-width: 960px;
            margin: 0 auto;
            padding: 0 1.5rem 2rem;
            font-size: 0.85rem;
            color: var(--muted);
        }}
        .meta-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(15, 23, 42, 0.2);
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.25s ease;
        }}
        body.meta-open .meta-overlay {{
            opacity: 1;
            pointer-events: auto;
        }}
        aside#meta-panel {{
            position: fixed;
            top: 0;
            right: 0;
            width: min(400px, 90vw);
            height: 100vh;
            background: var(--card);
            box-shadow: -24px 0 60px -40px rgba(15, 23, 42, 0.6);
            transform: translateX(100%);
            transition: transform 0.3s ease;
            display: flex;
            flex-direction: column;
            padding: 1.5rem;
            gap: 1rem;
            overflow-y: auto;
        }}
        body.meta-open aside#meta-panel {{
            transform: translateX(0);
        }}
        aside#meta-panel header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        aside#meta-panel h2 {{
            margin: 0;
            font-size: 1.2rem;
        }}
        aside#meta-panel dl {{
            margin: 0;
            display: grid;
            grid-template-columns: max-content 1fr;
            gap: 0.5rem 1.2rem;
            font-size: 0.9rem;
        }}
        aside#meta-panel dt {{
            font-weight: 600;
            color: var(--muted);
        }}
        aside#meta-panel dd {{
            margin: 0;
            word-break: break-word;
        }}
        button.meta-close {{
            border: none;
            background: transparent;
            color: var(--muted);
            font-size: 0.95rem;
            cursor: pointer;
        }}
        @media (max-width: 640px) {{
            header.page-header {{
                flex-direction: column;
                align-items: flex-start;
            }}
            button.meta-toggle {{
                align-self: stretch;
                text-align: center;
            }}
            .event summary {{
                grid-template-columns: 1fr;
                gap: 0.4rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="meta-overlay" id="meta-overlay"></div>
    <aside id="meta-panel" aria-hidden="true">
        <header>
            <h2>Run details</h2>
            <button class="meta-close" id="close-meta" type="button">Close</button>
        </header>
        <dl>{metadata_html}</dl>
    </aside>
    <header class="page-header">
        <h1>Run Visualizer</h1>
        <button class="meta-toggle" id="toggle-meta" type="button" aria-expanded="false" aria-controls="meta-panel">Show details</button>
    </header>
    <main>
        {render_text('Model input', sections.get('user_instructions', ''), collapsible=True, collapsed_by_default=True)}
        {render_text('Model response', sections.get('final_answer', ''))}
        {render_text('Correct answer', sections.get('correct_answer', ''), is_correct_answer=True) if sections.get('correct_answer') else ''}
        {tail_block}
        {image_control}
        {event_block}
    </main>
    <footer>Generated by visualize_run.py</footer>
    <script>
    // Visual picker
    const select = document.getElementById('image-select');
    const caption = document.getElementById('image-caption');
    const img = document.getElementById('preview');

    if (select && caption && img) {{
        const update = () => {{
            const option = (select.selectedOptions && select.selectedOptions[0]) || select.options[select.selectedIndex] || null;
            const label = option ? option.text : '';
            const value = select.value || '';
            img.src = value;
            img.alt = label || 'Selected visual';
            caption.textContent = label;
        }};
        select.addEventListener('change', update);
        update(); // initialize
    }}

    // Meta panel controls
    const body = document.body;
    const toggleBtn = document.getElementById('toggle-meta');
    const closeBtn = document.getElementById('close-meta');
    const overlay = document.getElementById('meta-overlay');
    const metaPanel = document.getElementById('meta-panel');

    const openPanel = () => {{
        body.classList.add('meta-open');
        toggleBtn.setAttribute('aria-expanded', 'true');
        metaPanel.setAttribute('aria-hidden', 'false');
    }};

    const closePanel = () => {{
        body.classList.remove('meta-open');
        toggleBtn.setAttribute('aria-expanded', 'false');
        metaPanel.setAttribute('aria-hidden', 'true');
    }};

    if (toggleBtn) {{
        toggleBtn.addEventListener('click', () => {{
            if (body.classList.contains('meta-open')) {{
                closePanel();
            }} else {{
                openPanel();
            }}
        }});
    }}

    if (closeBtn) {{
        closeBtn.addEventListener('click', closePanel);
    }}

    if (overlay) {{
        overlay.addEventListener('click', closePanel);
    }}

    document.addEventListener('keydown', (event) => {{
        if (event.key === 'Escape' && body.classList.contains('meta-open')) {{
            closePanel();
        }}
    }});
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
        sections, events = extract_sections(log_path)
        images = gather_images(asset_dirs)
        # Use the original top-level 'source' for metadata "Source", and the actual log path for "Log file"
        html_text = build_html(source, log_path, sections, events, images, output_path, asset_dirs)
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
