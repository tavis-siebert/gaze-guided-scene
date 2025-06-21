#!/usr/bin/env python3
"""Export Mermaid diagrams from markdown files to PDF or PNG format.

This script scans all .md files in the figures directory, extracts mermaid
diagram content, and exports each diagram to a PDF or PNG file in figures/out.
Supports transparent backgrounds (recommended for PNG format).
"""

import argparse
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
    mermaid_content: str,
    output_filename: str = "heterogeneous_graph_mermaid.pdf",
    background_color: str = "transparent",
    output_format: str = "pdf",
) -> bool:
    """Export Mermaid diagram to PDF/PNG using mermaid-cli.

    Args:
        mermaid_content: The mermaid diagram content
        output_filename: Output file path
        background_color: Background color ('white', 'transparent', or any valid color)
        output_format: Output format ('pdf' or 'png')
    """
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
            # Use mermaid-cli to convert to specified format
            cmd = [
                "mmdc",
                "-i",
                temp_file_path,
                "-o",
                output_filename,
                "-f",
                output_format,
                "--backgroundColor",
                background_color,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            print(f"✓ Successfully exported {output_format.upper()}: {output_filename}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"✗ Error exporting {output_format.upper()}: {e}")
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


def generate_output_filename(
    md_file: Path, diagram_index: int = 0, output_format: str = "pdf"
) -> str:
    """Generate output filename based on markdown file name and format."""
    base_name = md_file.stem
    extension = output_format.lower()
    if diagram_index > 0:
        return f"{base_name}_diagram_{diagram_index + 1}.{extension}"
    return f"{base_name}.{extension}"


def process_markdown_files(
    figures_dir: Path, output_dir: Path, background_color: str, output_format: str
) -> None:
    """Process all markdown files and export mermaid diagrams to specified format."""
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
            output_filename = generate_output_filename(md_file, i, output_format)
            output_path = output_dir / output_filename

            print(f"  Exporting diagram {i + 1} to {output_filename}")

            if export_to_pdf(
                mermaid_content, str(output_path), background_color, output_format
            ):
                successful_exports += 1

    print(f"\n{'=' * 50}")
    print("Export Summary:")
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

    parser = argparse.ArgumentParser(
        description="Export Mermaid diagrams to PDF or PNG"
    )
    parser.add_argument(
        "--background-color",
        type=str,
        default="transparent",
        help="Background color for the diagrams (e.g., 'white', 'transparent', '#ffffff')",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="pdf",
        choices=["pdf", "png"],
        help="Output format for the diagrams (pdf or png)",
    )
    args = parser.parse_args()

    process_markdown_files(
        figures_dir, output_dir, args.background_color, args.output_format
    )


if __name__ == "__main__":
    main()
