#!/usr/bin/env python3
from pathlib import Path
import shutil
import re
import yaml
import fire
from rich.console import Console
from rich.progress import track
from rich.panel import Panel

console = Console()

def generate_docs_site(output_dir: str = "archivum_site"):
    """
    Generate a MkDocs site from Archivum content.
    
    Args:
        output_dir: Directory to create the MkDocs files in
    """
    root_dir = Path.cwd()
    output_path = Path(output_dir)
    
    # Create docs directory structure
    docs_dir = output_path / "docs"
    if docs_dir.exists():
        console.print(f"[yellow]Warning: Output directory {docs_dir} already exists. Files may be overwritten.[/]")
    
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # First, let's organize the navigation structure
    nav_structure = []
    
    # Copy README as index
    readme_path = root_dir / "README.md"
    if readme_path.exists():
        shutil.copy(readme_path, docs_dir / "index.md")
        nav_structure.append({"Home": "index.md"})
        console.print(f"Copied: [green]README.md[/] -> [blue]index.md[/]")
    
    # Create a directory for our assets
    assets_dir = docs_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    
    # Process directories numerically
    dirs = [d for d in root_dir.iterdir() if d.is_dir() and re.match(r'^\d+_', d.name)]
    
    for dir_path in track(sorted(dirs), description="Processing directories"):
        dir_name = dir_path.name
        
        # Extract number and name
        if '_' in dir_name:
            num, name = dir_name.split('_', 1)
            # Make directory name more readable
            clean_name = name.replace('_', ' ').title()
        else:
            clean_name = dir_name
            name = dir_name
            
        section_dir = docs_dir / name
        section_dir.mkdir(exist_ok=True)
        
        # Create a section in the navigation
        section_nav = {clean_name: []}
        
        # Copy Markdown files
        md_files = list(dir_path.glob("**/*.md"))
        
        if not md_files:
            # If no Markdown files, add a simple entry
            section_nav[clean_name] = f"{name}/index.md"
            # Create a placeholder index file
            index_content = f"# {clean_name}\n\nThis section contains no Markdown files yet."
            (section_dir / "index.md").write_text(index_content)
        else:
            # Process all Markdown files
            for md_file in md_files:
                # Get relative path from the source directory
                rel_path = md_file.relative_to(dir_path)
                target_path = section_dir / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy and process the Markdown file
                process_md_file(md_file, target_path)
                
                # Add to navigation (using path relative to docs/)
                nav_path = str(target_path.relative_to(docs_dir))
                
                # Get title from the file
                title = get_title_from_md(md_file)
                
                # If this is a README or similar index file at the root, make it the section index
                if rel_path.name.lower() in ("readme.md", "index.md") and len(rel_path.parts) == 1:
                    # Make it the section landing page
                    section_nav[clean_name] = nav_path
                else:
                    # Only add if section_nav[clean_name] is still a list
                    if isinstance(section_nav[clean_name], list):
                        section_nav[clean_name].append({title: nav_path})
        
        # Add this section to the main navigation
        nav_structure.append(section_nav)
    
    # Generate mkdocs.yml with our navigation
    generate_mkdocs_config(output_path, nav_structure)
    
    console.print(Panel.fit(
        f"[bold green]MkDocs site generated in {output_path}[/]\n\n"
        f"To preview the site, run:\n"
        f"[bold]cd {output_path} && mkdocs serve[/]\n\n"
        f"To install MkDocs if needed:\n"
        f"[bold]pip install mkdocs mkdocs-material[/]",
        title="Success", 
        border_style="green"
    ))

def get_title_from_md(file_path: Path) -> str:
    """Extract a title from a Markdown file, using frontmatter title or first heading."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Check for YAML frontmatter
        if content.startswith('---'):
            try:
                # Extract YAML frontmatter
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    if frontmatter and 'title' in frontmatter:
                        return frontmatter['title'].strip('"\'')
                    elif frontmatter and 'project_name' in frontmatter:
                        return frontmatter['project_name'].strip('"\'')
            except Exception:
                # If YAML parsing fails, continue to look for headings
                pass
        
        # Look for the first heading
        heading_match = re.search(r'^#\s+(.+?)$', content, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()
            
        # Fall back to the filename
        return file_path.stem.replace('_', ' ').title()
    except Exception:
        return file_path.stem.replace('_', ' ').title()

def process_md_file(source_path: Path, target_path: Path):
    """
    Process a Markdown file, potentially modifying it for MkDocs.
    
    This includes:
    - Handling YAML frontmatter (converting to proper format or removing)
    - Replacing template variables like {{project_name}}
    - Adjusting internal links
    - Fixing image references
    """
    try:
        content = source_path.read_text(encoding='utf-8')
        
        # Process YAML frontmatter if present
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    # Load frontmatter as YAML
                    frontmatter = yaml.safe_load(parts[1])
                    
                    # Extract useful metadata for display
                    metadata_section = ""
                    if frontmatter:
                        metadata_section = "## Metadata\n\n"
                        for key, value in frontmatter.items():
                            if key in ['title', 'project_name', 'date', 'status', 'tags', 'lead']:
                                if isinstance(value, list):
                                    value_str = ", ".join([str(v) for v in value])
                                    metadata_section += f"- **{key}**: {value_str}\n"
                                else:
                                    metadata_section += f"- **{key}**: {value}\n"
                        metadata_section += "\n"
                    
                    # Use the content without frontmatter, adding metadata section
                    content = parts[2]
                    
                    # Replace template variables with their actual values
                    template_vars = {
                        '{{project_name}}': frontmatter.get('project_name', ''),
                        '{{title}}': frontmatter.get('title', ''),
                        '{{date}}': frontmatter.get('date', '')
                    }
                    
                    for template_var, value in template_vars.items():
                        if value:  # Only replace if we have a value
                            content = content.replace(template_var, value)
                    
                    # If the content doesn't start with a heading, add title from frontmatter
                    if not re.match(r'^#\s+', content.lstrip()):
                        title = frontmatter.get('title', frontmatter.get('project_name', source_path.stem))
                        content = f"# {title}\n\n{metadata_section}{content}"
                    else:
                        # Insert metadata after the first heading
                        content = re.sub(r'^(#\s+.+?\n\n)', r'\1' + metadata_section, content, 1, re.DOTALL)
                        
                except Exception as e:
                    # If YAML parsing fails, leave as is
                    console.print(f"[yellow]Warning: YAML parsing failed for {source_path}: {e}[/]")
        
        # Write processed content
        target_path.write_text(content, encoding='utf-8')
        console.print(f"Processed: [green]{source_path}[/] -> [blue]{target_path}[/]")
    except Exception as e:
        console.print(f"[bold red]Error processing {source_path}:[/] {e}")

def generate_mkdocs_config(output_path: Path, nav_structure: list):
    """Generate the MkDocs configuration file with navigation."""
    config = {
        'site_name': 'Archivum',
        'site_description': 'An LLM-Assisted Personal Knowledge & Project Management System',
        'theme': {
            'name': 'material',
            'palette': {
                'primary': 'indigo',
                'accent': 'indigo'
            },
            'features': [
                'navigation.instant',
                'navigation.tracking',
                'navigation.expand',
                'navigation.indexes',
                'navigation.top',
                'search.highlight',
                'search.share',
                'search.suggest'
            ]
        },
        'plugins': [
            'search',
        ],
        'markdown_extensions': [
            'admonition',
            'pymdownx.details',
            'pymdownx.superfences',
            'pymdownx.highlight',
            'pymdownx.inlinehilite',
            'pymdownx.tabbed',
            'pymdownx.tasklist',
            'toc'
        ],
        'nav': nav_structure
    }
    
    # Write the YAML configuration
    with open(output_path / "mkdocs.yml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    fire.Fire(generate_docs_site) 