from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter


def assemble_pdf_suite(input_pdfs: list[Path], output_pdf: Path) -> None:
    writer = PdfWriter()
    for input_pdf in input_pdfs:
        reader = PdfReader(str(input_pdf))
        for page in reader.pages:
            writer.add_page(page)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as handle:
        writer.write(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple PDFs into a single suite PDF.")
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--input-pdf", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assemble_pdf_suite(args.input_pdf, args.output_pdf)
    print(f"Wrote PDF suite: {args.output_pdf}")


if __name__ == "__main__":
    main()
