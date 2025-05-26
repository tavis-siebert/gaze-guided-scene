#!/usr/bin/env python3
"""Export Mermaid diagrams from markdown files to PDF format.

This script scans all .md files in the figures directory, extracts mermaid
diagram content, and exports each diagram to a PDF file in figures/out.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List


def extract_mermaid_blocks(markdown_content: str) -> List[str]:
    """Extract all mermaid code blocks from markdown content."""
    pattern = r"```mermaid\n(.*?)\n```"
    matches = re.findall(pattern, markdown_content, re.DOTALL)
    return matches


def export_to_pdf(
    mermaid_content: str, output_filename: str = "heterogeneous_graph_mermaid.pdf"
) -> bool:
    """Export Mermaid diagram to PDF using mermaid-cli."""
    try:
        # Create temporary file with mermaid content
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mmd", delete=False
        ) as temp_file:
            # Extract just the mermaid code without markdown backticks
            mermaid_code = mermaid_content.replace("```mermaid\n", "").replace(
                "\n```", ""
            )
            temp_file.write(mermaid_code)
            temp_file_path = temp_file.name

        try:
            # Use mermaid-cli to convert to PDF
            result = subprocess.run(
                [
                    "mmdc",
                    "-i",
                    temp_file_path,
                    "-o",
                    output_filename,
                    "-f",
                    "pdf",
                    "--backgroundColor",
                    "white",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            print(f"✓ Successfully exported PDF: {output_filename}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"✗ Error exporting PDF: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
            print(
                "  Make sure mermaid-cli is installed: npm install -g @mermaid-js/mermaid-cli"
            )
            return False

        except FileNotFoundError:
            print("✗ mermaid-cli (mmdc) not found.")
            print("  Install with: npm install -g @mermaid-js/mermaid-cli")
            return False

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def find_markdown_files(figures_dir: Path) -> List[Path]:
    """Find all .md files in the figures directory."""
    return list(figures_dir.glob("*.md"))


def generate_output_filename(md_file: Path, diagram_index: int = 0) -> str:
    """Generate output PDF filename based on markdown file name."""
    base_name = md_file.stem
    if diagram_index > 0:
        return f"{base_name}_diagram_{diagram_index + 1}.pdf"
    return f"{base_name}.pdf"


def process_markdown_files(figures_dir: Path, output_dir: Path) -> None:
    """Process all markdown files and export mermaid diagrams to PDFs."""
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    md_files = find_markdown_files(figures_dir)

    if not md_files:
        print(f"No markdown files found in {figures_dir}")
        return

    total_diagrams = 0
    successful_exports = 0

    for md_file in md_files:
        print(f"\nProcessing: {md_file.name}")

        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"✗ Error reading {md_file}: {e}")
            continue

        mermaid_blocks = extract_mermaid_blocks(content)

        if not mermaid_blocks:
            print(f"  No mermaid diagrams found in {md_file.name}")
            continue

        print(f"  Found {len(mermaid_blocks)} mermaid diagram(s)")

        for i, mermaid_content in enumerate(mermaid_blocks):
            total_diagrams += 1
            output_filename = generate_output_filename(md_file, i)
            output_path = output_dir / output_filename

            print(f"  Exporting diagram {i + 1} to {output_filename}")

            if export_to_pdf(mermaid_content, str(output_path)):
                successful_exports += 1

    print(f"\n{'=' * 50}")
    print(f"Export Summary:")
    print(f"  Total diagrams found: {total_diagrams}")
    print(f"  Successfully exported: {successful_exports}")
    print(f"  Failed exports: {total_diagrams - successful_exports}")
    print(f"  Output directory: {output_dir}")


def main():
    """Main entry point."""
    # Get script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    figures_dir = project_root / "figures"
    output_dir = figures_dir / "out"

    if not figures_dir.exists():
        print(f"✗ Figures directory not found: {figures_dir}")
        return

    print(f"Scanning for markdown files in: {figures_dir}")
    print(f"Output directory: {output_dir}")

    process_markdown_files(figures_dir, output_dir)


if __name__ == "__main__":
    main()
