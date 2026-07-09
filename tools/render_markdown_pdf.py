from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def _escape_latex(text: str) -> str:
    escaped = (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )
    return escaped


def _convert_inline_markdown(text: str) -> str:
    token_pattern = re.compile(r"`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)")
    parts: list[str] = []
    last_end = 0
    for match in token_pattern.finditer(text):
        if match.start() > last_end:
            parts.append(_escape_latex(text[last_end:match.start()]))
        if match.group(1) is not None:
            parts.append(r"\texttt{" + _escape_latex(match.group(1)) + "}")
        else:
            label = _escape_latex(match.group(2))
            target = match.group(3)
            parts.append(rf"\href{{{target}}}{{{label}}}")
        last_end = match.end()
    if last_end < len(text):
        parts.append(_escape_latex(text[last_end:]))
    return "".join(parts)


def markdown_to_latex(markdown_text: str, title: str) -> str:
    lines = markdown_text.splitlines()
    body: list[str] = []
    in_itemize = False
    in_enumerate = False

    def close_lists() -> None:
        nonlocal in_itemize, in_enumerate
        if in_itemize:
            body.append(r"\end{itemize}")
            in_itemize = False
        if in_enumerate:
            body.append(r"\end{enumerate}")
            in_enumerate = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            close_lists()
            body.append("")
            continue

        if stripped.startswith("# "):
            close_lists()
            body.append(rf"\section*{{{_convert_inline_markdown(stripped[2:].strip())}}}")
            continue
        if stripped.startswith("## "):
            close_lists()
            body.append(rf"\subsection*{{{_convert_inline_markdown(stripped[3:].strip())}}}")
            continue
        if stripped.startswith("### "):
            close_lists()
            body.append(rf"\subsubsection*{{{_convert_inline_markdown(stripped[4:].strip())}}}")
            continue
        if stripped.startswith("- "):
            if in_enumerate:
                body.append(r"\end{enumerate}")
                in_enumerate = False
            if not in_itemize:
                body.append(r"\begin{itemize}")
                in_itemize = True
            body.append(rf"\item {_convert_inline_markdown(stripped[2:].strip())}")
            continue
        if re.match(r"^\d+\.\s+", stripped):
            if in_itemize:
                body.append(r"\end{itemize}")
                in_itemize = False
            if not in_enumerate:
                body.append(r"\begin{enumerate}")
                in_enumerate = True
            item_text = re.sub(r"^\d+\.\s+", "", stripped, count=1)
            body.append(rf"\item {_convert_inline_markdown(item_text)}")
            continue
        if stripped.startswith("> "):
            close_lists()
            body.append(r"\begin{quote}")
            body.append(_convert_inline_markdown(stripped[2:].strip()))
            body.append(r"\end{quote}")
            continue

        close_lists()
        body.append(_convert_inline_markdown(stripped))

    close_lists()

    document = rf"""
\documentclass[11pt]{{ctexart}}
\usepackage[a4paper,margin=2.2cm]{{geometry}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\setlist[itemize]{{leftmargin=1.8em}}
\setlist[enumerate]{{leftmargin=1.8em}}
\title{{{_escape_latex(title)}}}
\date{{}}
\begin{{document}}
\maketitle
{chr(10).join(body)}
\end{{document}}
""".strip()
    return document


def render_markdown_pdf(markdown_path: Path, output_pdf: Path, title: str | None = None) -> None:
    markdown_text = markdown_path.read_text(encoding="utf-8")
    chosen_title = title or markdown_path.stem
    latex_text = markdown_to_latex(markdown_text, chosen_title)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        tex_path = temp_dir_path / "document.tex"
        tex_path.write_text(latex_text, encoding="utf-8")
        for _ in range(2):
            subprocess.run(
                [
                    "xelatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    str(tex_path.name),
                ],
                cwd=str(temp_dir_path),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        generated_pdf = temp_dir_path / "document.pdf"
        output_pdf.write_bytes(generated_pdf.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Markdown report to PDF using XeLaTeX.")
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--title", help="Optional explicit document title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_markdown_pdf(args.markdown, args.output_pdf, title=args.title)
    print(f"Wrote PDF: {args.output_pdf}")


if __name__ == "__main__":
    main()
