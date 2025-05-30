#!/usr/bin/env python3
import fire
from pypdf import PdfReader
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

def extract_text(pdf_path: str):
    """
    Extracts text from a PDF file and prints it to the console.

    Args:
        pdf_path: The path (absolute or relative) to the PDF file.
    """
    try:
        # Convert to Path object
        pdf_path = Path(pdf_path)
        
        # If not absolute, resolve to absolute path
        if not pdf_path.is_absolute():
            pdf_path = pdf_path.resolve()

        console.print(f"[bold blue]Attempting to read PDF from:[/] {pdf_path}")

        if not pdf_path.exists():
            console.print(f"[bold red]Error:[/] PDF file not found at: {pdf_path}")
            return

        reader = PdfReader(pdf_path)
        extracted_text = ""
        console.print(f"[bold green]Found {len(reader.pages)} pages.[/]")

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text += f"\n\n--- Page {i+1} ---\n"
                extracted_text += page_text
            else:
                extracted_text += f"\n\n--- Page {i+1} (No text extracted) ---\n"

        console.print(Panel.fit("[bold]Extracted Text[/]", border_style="blue"))
        # Use Syntax to display with potential highlighting
        console.print(Syntax(extracted_text, "text", background_color="default"))
        console.print(Panel.fit("[bold]End of Extracted Text[/]", border_style="blue"))

    except Exception as e:
        console.print(f"[bold red]An error occurred:[/] {e}")

if __name__ == '__main__':
    fire.Fire(extract_text) 