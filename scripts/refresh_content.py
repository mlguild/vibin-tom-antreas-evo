#!/usr/bin/env python3
import glob
from datetime import datetime, timedelta
from pathlib import Path
import fire
from rich.console import Console
from rich.progress import track

console = Console()

def read_file(file_path: Path) -> str:
    """Read and return the contents of a file with error handling."""
    try:
        content = file_path.read_text(encoding='utf-8')
        return f"==== CONTENT OF {file_path} ====\n\n{content}\n\n"
    except Exception as e:
        return f"==== ERROR READING {file_path} ====\n{str(e)}\n\n"

def refresh_content(output_file: str = "archivum_content.txt"):
    """
    Combines content from various files in the Archivum repository into a single file.
    
    Args:
        output_file: The filename to save the combined content to.
    """
    # Get the current directory (assuming script is run from the project root)
    root_dir = Path.cwd()
    
    output = []
    
    # 1. Read core files
    console.print("[bold blue]Reading core files...[/]")
    core_files = [
        "README.md",
        "config.yaml",
        ".cursorrules"
    ]
    for file in core_files:
        file_path = root_dir / file
        if file_path.exists():
            output.append(read_file(file_path))
    
    # 2. Read all templates
    console.print("[bold blue]Reading templates...[/]")
    templates_dir = root_dir / "00_meta" / "templates"
    if templates_dir.exists():
        template_files = list(templates_dir.glob("*.md"))
        for file in track(template_files, description="Processing templates"):
            output.append(read_file(file))
    
    # 3. Read all scripts
    console.print("[bold blue]Reading scripts...[/]")
    scripts_dir = root_dir / "00_meta" / "scripts"
    if scripts_dir.exists():
        script_files = list(scripts_dir.glob("*.py"))
        for file in track(script_files, description="Processing scripts"):
            output.append(read_file(file))
    
    # 4. Read master todo list
    console.print("[bold blue]Reading master todo list...[/]")
    todo_file = root_dir / "04_global_tasks_and_planning" / "master_todo.md"
    if todo_file.exists():
        output.append(read_file(todo_file))
    
    # 5. Read all project READMEs
    console.print("[bold blue]Reading project READMEs...[/]")
    projects_dir = root_dir / "01_projects"
    if projects_dir.exists():
        project_dirs = [d for d in projects_dir.iterdir() if d.is_dir()]
        for project_dir in track(project_dirs, description="Processing projects"):
            readme_path = project_dir / "README.md"
            if readme_path.exists():
                output.append(read_file(readme_path))
    
    # 6. Read files from other directories
    directories = [
        "02_ideas_and_explorations",
        "03_knowledge_base",
        "05_outputs_and_dissemination",
        "06_reviews_and_service",
        "07_professional_development",
        "10_philosophy_and_reflection"
    ]
    
    for directory in directories:
        console.print(f"[bold blue]Reading files from {directory}...[/]")
        dir_path = root_dir / directory
        if dir_path.exists():
            md_files = list(dir_path.glob("*.md"))
            for file in track(md_files, description=f"Processing {directory}"):
                output.append(read_file(file))
    
    # 7. Read recent daily journal entries (last 7 days)
    console.print("[bold blue]Reading recent daily journal entries...[/]")
    journal_dir = root_dir / "08_daily_journal"
    if journal_dir.exists():
        today = datetime.now()
        for i in range(7):
            date = today - timedelta(days=i)
            year_month = date.strftime("%Y/%m")
            year_month_dir = journal_dir / year_month
            if year_month_dir.exists():
                day_file = year_month_dir / f"{date.strftime('%Y-%m-%d')}.md"
                if day_file.exists():
                    output.append(read_file(day_file))
    
    # 8. Read current week calendar entries
    console.print("[bold blue]Reading current week calendar entries...[/]")
    calendar_dir = root_dir / "09_calendar"
    if calendar_dir.exists():
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        for i in range(7):
            date = start_of_week + timedelta(days=i)
            year_month = date.strftime("%Y/%m")
            year_month_dir = calendar_dir / year_month
            if year_month_dir.exists():
                day_file = year_month_dir / f"{date.strftime('%Y-%m-%d')}.md"
                if day_file.exists():
                    output.append(read_file(day_file))
    
    # 9. Read personal journal entries (with discretion)
    console.print("[bold blue]Reading personal journal entries...[/]")
    personal_dir = root_dir / "11_personal_journal"
    if personal_dir.exists():
        personal_files = list(personal_dir.glob("*.md"))
        for file in track(personal_files, description="Processing personal journal"):
            output.append(read_file(file))
    
    # Write combined output to file
    out_file_path = root_dir / output_file
    out_file_path.write_text("\n".join(output), encoding='utf-8')
    
    console.print(f"[bold green]Content has been combined and saved to {out_file_path}[/]")
    console.print(f"[bold green]Total files processed: {len(output)}[/]")

if __name__ == "__main__":
    fire.Fire(refresh_content) 