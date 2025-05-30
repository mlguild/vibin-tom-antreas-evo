#!/usr/bin/env python3
import fire
from pathlib import Path
import yaml
from datetime import datetime
import re
import subprocess
from rich.console import Console
from rich.progress import track

console = Console()

# Directories to scan within the Archivum
ARCHIVUM_DIRS_TO_SCAN = [
    "00_meta",  # Templates might also benefit from accurate dates if they evolve
    "01_projects",
    "02_ideas_and_explorations",
    "03_knowledge_base",
    "04_global_tasks_and_planning",
    "05_outputs_and_dissemination",
    "06_reviews_and_service",
    "07_professional_development",
    "08_daily_journal",
    "09_calendar",
    "10_philosophy_and_reflection",
    "11_personal_journal",
    "12_meetings_and_notes",
]

# Placeholder for dates in templates
DATE_PLACEHOLDER = "YYYY-MM-DD"


def is_template_file(file_path: Path) -> bool:
    """Check if a file is a template that should keep placeholder dates."""
    return "templates" in file_path.parts


def get_git_first_commit_date(file_path: Path) -> str:
    """Get the date of the first git commit for a file."""
    try:
        # Get the first commit date for this file
        result = subprocess.run(
            [
                "git",
                "log",
                "--follow",
                "--format=%ad",
                "--date=short",
                "--",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            cwd=file_path.parent if file_path.parent.exists() else Path.cwd(),
        )

        if result.returncode == 0 and result.stdout.strip():
            # Get the last line (oldest commit)
            commit_dates = result.stdout.strip().split("\n")
            if commit_dates:
                return commit_dates[-1]  # Last line is the oldest commit
    except Exception as e:
        console.print(
            f"[yellow]Could not get git commit date for {file_path.name}: {e}[/yellow]"
        )

    return None


def is_valid_date_str(date_str):
    """Check if a string is a valid date and not the placeholder."""
    if not date_str or date_str == DATE_PLACEHOLDER:
        return False
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def compare_dates(date1_str: str, date2_str: str) -> int:
    """Compare two date strings. Returns -1 if date1 < date2, 0 if equal, 1 if date1 > date2."""
    try:
        date1 = datetime.strptime(date1_str, "%Y-%m-%d")
        date2 = datetime.strptime(date2_str, "%Y-%m-%d")
        if date1 < date2:
            return -1
        elif date1 > date2:
            return 1
        else:
            return 0
    except ValueError:
        return 0


def update_markdown_entry_dates(file_path: Path, today_str: str):
    """Updates created_date and last_updated_date in a single Markdown file."""
    try:
        # Skip template files - they should keep placeholder dates
        if is_template_file(file_path):
            return False

        content = file_path.read_text(encoding="utf-8")

        if not content.startswith("---"):
            return False

        parts = content.split("---", 2)
        if len(parts) < 3:
            return False

        frontmatter_str = parts[1]
        main_content = parts[2]

        try:
            data = yaml.safe_load(frontmatter_str)
            if data is None:
                data = {}
        except yaml.YAMLError as e:
            console.print(
                f"[red]Error parsing YAML for {file_path.name}: {e}[/red]"
            )
            return False

        original_data_str = str(data)
        modified = False

        stat_info = file_path.stat()
        m_timestamp = stat_info.st_mtime
        b_timestamp = getattr(stat_info, "st_birthtime", None)
        formatted_mtime = datetime.fromtimestamp(m_timestamp).strftime(
            "%Y-%m-%d"
        )
        formatted_btime = (
            datetime.fromtimestamp(b_timestamp).strftime("%Y-%m-%d")
            if b_timestamp
            else None
        )

        # Get git first commit date
        git_first_commit = get_git_first_commit_date(file_path)

        action_log = []

        if data.get("last_updated_date") != today_str:
            data["last_updated_date"] = today_str
            action_log.append(f"Set last_updated_date to {today_str}")
            modified = True

        current_created_date = data.get("created_date")
        if not is_valid_date_str(current_created_date):
            date_field_val = data.get("date")
            created_date_candidate = None
            source_log = ""

            if is_valid_date_str(date_field_val):
                created_date_candidate = date_field_val
                source_log = f"from 'date' field: {date_field_val}"
            elif git_first_commit:
                created_date_candidate = git_first_commit
                source_log = f"from git first commit: {git_first_commit}"
            elif formatted_btime:
                created_date_candidate = formatted_btime
                source_log = f"from file birth time: {formatted_btime}"
            else:
                created_date_candidate = formatted_mtime
                source_log = f"from file modification time: {formatted_mtime}"

            # Ensure created_date is not older than git first commit
            if (
                git_first_commit
                and compare_dates(created_date_candidate, git_first_commit) < 0
            ):
                created_date_candidate = git_first_commit
                source_log = f"adjusted to git first commit (was older): {git_first_commit}"

            if data.get("created_date") != created_date_candidate:
                data["created_date"] = created_date_candidate
                action_log.append(f"Set created_date {source_log}")
                modified = True
        else:
            # Check if existing created_date is older than git first commit
            if (
                git_first_commit
                and compare_dates(current_created_date, git_first_commit) < 0
            ):
                data["created_date"] = git_first_commit
                action_log.append(
                    f"Updated created_date to git first commit (was older): {git_first_commit}"
                )
                modified = True
            elif not action_log:
                action_log.append(
                    f"Kept existing created_date: {current_created_date}"
                )

        if "finished_date" not in data:
            data["finished_date"] = DATE_PLACEHOLDER
            action_log.append(f"Added placeholder for finished_date.")
            modified = True

        if not modified and data.get("last_updated_date") == formatted_mtime:
            if original_data_str == str(data):
                return False

        updated_frontmatter_block = yaml.dump(
            data, sort_keys=False, default_flow_style=False, allow_unicode=True
        ).strip()

        new_content = f"""---
{updated_frontmatter_block}
---
{main_content.lstrip()}"""

        file_path.write_text(new_content, encoding="utf-8")
        if (
            action_log
        ):  # Only print if actions were logged (meaning a change was made or considered)
            console.print(
                f"[green]Updated {file_path.name}[/green]: {'; '.join(action_log)}"
            )
        return True

    except Exception as e:
        console.print(
            f"[bold red]Failed to process {file_path.name}: {e}[/bold red]"
        )
        return False


def update_all_entries(dry_run: bool = False):
    root_dir = Path.cwd()
    today_str = datetime.now().strftime("%Y-%m-%d")
    updated_files_count = 0
    processed_files_count = 0
    skipped_templates_count = 0

    console.print(
        f"[bold blue]Starting date update process. Today's date: {today_str}[/]"
    )
    if dry_run:
        console.print(
            "[bold yellow]DRY RUN enabled. No files will be modified.[/bold yellow]"
        )

    all_md_files = []
    for dir_name in ARCHIVUM_DIRS_TO_SCAN:
        scan_path = root_dir / dir_name
        if scan_path.is_dir():
            if dir_name == "archivum_site":
                console.print(
                    f"[dim]Skipping {dir_name} directory intentionally.[/dim]"
                )
                continue
            all_md_files.extend(list(scan_path.rglob("*.md")))
        else:
            console.print(
                f"[yellow]Directory not found, skipping: {scan_path}[/yellow]"
            )

    if not all_md_files:
        console.print(
            "[yellow]No Markdown files found in the specified directories.[/yellow]"
        )
        return

    all_md_files = [
        f
        for f in all_md_files
        if ".git" not in [part.lower() for part in f.parts]
    ]

    for file_path in track(
        all_md_files, description="Updating Markdown entries"
    ):
        processed_files_count += 1

        # Skip template files
        if is_template_file(file_path):
            skipped_templates_count += 1
            if dry_run:
                console.print(
                    f"[dim][DRY RUN] Skipping template: {file_path.name}[/dim]"
                )
            continue

        if dry_run:
            try:
                content = file_path.read_text(encoding="utf-8")
                if not content.startswith("---"):
                    continue
                parts = content.split("---", 2)
                if len(parts) < 3:
                    continue
                data = yaml.safe_load(parts[1]) or {}

                stat_info = file_path.stat()
                m_timestamp = stat_info.st_mtime
                b_timestamp = getattr(stat_info, "st_birthtime", None)
                formatted_mtime = datetime.fromtimestamp(m_timestamp).strftime(
                    "%Y-%m-%d"
                )
                formatted_btime = (
                    datetime.fromtimestamp(b_timestamp).strftime("%Y-%m-%d")
                    if b_timestamp
                    else None
                )

                # Get git first commit date for dry run
                git_first_commit = get_git_first_commit_date(file_path)

                dry_action_log = []
                if data.get("last_updated_date") != today_str:
                    dry_action_log.append(
                        f"Would set last_updated_date to {today_str}"
                    )

                current_created_date = data.get("created_date")
                created_date_updated = False
                if not is_valid_date_str(current_created_date):
                    date_field_val = data.get("date")
                    created_date_candidate = None
                    source_log = ""
                    if is_valid_date_str(date_field_val):
                        created_date_candidate = date_field_val
                        source_log = f"from 'date' field: {date_field_val}"
                    elif git_first_commit:
                        created_date_candidate = git_first_commit
                        source_log = (
                            f"from git first commit: {git_first_commit}"
                        )
                    elif formatted_btime:
                        created_date_candidate = formatted_btime
                        source_log = f"from file birth time: {formatted_btime}"
                    else:
                        created_date_candidate = formatted_mtime
                        source_log = (
                            f"from file modification time: {formatted_mtime}"
                        )

                    # Check git constraint in dry run
                    if (
                        git_first_commit
                        and compare_dates(
                            created_date_candidate, git_first_commit
                        )
                        < 0
                    ):
                        created_date_candidate = git_first_commit
                        source_log = f"adjusted to git first commit (was older): {git_first_commit}"

                    if data.get("created_date") != created_date_candidate:
                        dry_action_log.append(
                            f"Would set created_date {source_log}"
                        )
                        created_date_updated = True
                else:
                    # Check existing created_date against git in dry run
                    if (
                        git_first_commit
                        and compare_dates(
                            current_created_date, git_first_commit
                        )
                        < 0
                    ):
                        dry_action_log.append(
                            f"Would update created_date to git first commit (was older): {git_first_commit}"
                        )
                        created_date_updated = True

                if (
                    not created_date_updated
                    and not data.get("last_updated_date") == today_str
                ):  # only log if created_date wasn't logged as changed
                    if is_valid_date_str(current_created_date):
                        dry_action_log.append(
                            f"Would keep existing created_date: {current_created_date}"
                        )

                if "finished_date" not in data:
                    dry_action_log.append(
                        f"Would add placeholder for finished_date."
                    )

                if dry_action_log:  # Only print if there are actions
                    console.print(
                        f"[DRY RUN] {file_path.name}: {'; '.join(dry_action_log)}"
                    )

            except Exception as e:
                console.print(
                    f"[DRY RUN][bold red]Error simulating for {file_path.name}: {e}[/bold red]"
                )

        else:
            if update_markdown_entry_dates(file_path, today_str):
                updated_files_count += 1

    console.print(f"[bold green]Date update process finished.[/]")
    console.print(f"Processed {processed_files_count} Markdown files.")
    if skipped_templates_count > 0:
        console.print(
            f"[dim]Skipped {skipped_templates_count} template files.[/dim]"
        )
    if not dry_run:
        console.print(
            f"Updated {updated_files_count} files with new date information."
        )


if __name__ == "__main__":
    fire.Fire(update_all_entries)
