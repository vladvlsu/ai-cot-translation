import os
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.table import Table
from deep_translator import GoogleTranslator, MyMemoryTranslator
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from docx import Document
from striprtf.striprtf import rtf_to_text
import re
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Initialize Rich console
console = Console()

class TranslationManager:
    def __init__(self):
        self.console = Console()
        self.translators = {}
        
    def initialize_translator(self, direction):
        """Initialize the translator based on direction."""
        if direction == "rus2eng":
            self.translators[direction] = GoogleTranslator(source='ru', target='en')
        else:  # eng2rus
            self.translators[direction] = GoogleTranslator(source='en', target='ru')
            
    def translate_text(self, text, direction, progress, task_id):
        """Translate text with detailed progress tracking."""
        if direction not in self.translators:
            show_temp_status(progress, task_id, "🔄 Initializing translator...")
            self.initialize_translator(direction)
            
        translator = self.translators[direction]
        chunks = split_text_into_chunks(text)
        translated_chunks = []
        
        total_chunks = len(chunks)
        progress.update(task_id, total=total_chunks * 3)  # 3 steps per chunk
        
        for i, chunk in enumerate(chunks, 1):
            try:
                # Step 1: Pre-processing
                show_temp_status(progress, task_id, f"📝 Processing chunk {i}/{total_chunks}: Preparing...")
                progress.update(task_id, advance=1)
                
                # Step 2: Translation
                show_temp_status(progress, task_id, f"🔄 Processing chunk {i}/{total_chunks}: Translating...")
                translated_text = translator.translate(chunk)
                progress.update(task_id, advance=1)
                
                # Step 3: Post-processing
                show_temp_status(progress, task_id, f"✨ Processing chunk {i}/{total_chunks}: Finalizing...")
                translated_chunks.append(translated_text)
                progress.update(task_id, advance=1)
                
            except Exception as e:
                console.print(f"[red]Error translating chunk {i}: {str(e)}[/red]")
                # Fallback to MyMemory translator if Google fails
                try:
                    fallback_translator = MyMemoryTranslator(
                        source='ru' if direction == 'rus2eng' else 'en',
                        target='en' if direction == 'rus2eng' else 'ru'
                    )
                    translated_text = fallback_translator.translate(chunk)
                    translated_chunks.append(translated_text)
                except:
                    translated_chunks.append(chunk)  # Keep original if both translators fail
                continue
            
        return ' '.join(translated_chunks)

def split_text_into_chunks(text, max_chunk_size=1000):
    """Split text into chunks at sentence boundaries."""
    # Split text into sentences (basic implementation)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_translation_direction():
    """Get translation direction using arrow-key selection."""
    options = [
        ("🇷🇺 → 🇬🇧  Russian to English", "rus2eng"),
        ("🇬🇧 → 🇷🇺  English to Russian", "eng2rus")
    ]
    
    table = Table(show_header=False, show_edge=False, box=None)
    table.add_column("Direction")
    
    for label, value in options:
        table.add_row(f"  {label}")
    
    console.print("\n[bold blue]Select translation direction:[/bold blue]")
    console.print(table)
    
    choice = Prompt.ask(
        "Choose direction",
        choices=["1", "2"],
        default="1"
    )
    
    return options[int(choice) - 1][1]

def initialize_model():
    """Initialize the DRT model and tokenizer."""
    console.print(Panel("🤖 Initializing DRT-o1-14B model...", style="blue"))
    
    model_name = "Krystalan/DRT-o1-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def show_temp_status(progress, task_id, message, duration=5, completed_steps=None):
    """Show temporary status message for 5 seconds."""
    try:
        # Safely get task
        task = None
        for t in progress.tasks:
            if t.id == task_id:
                task = t
                break
                
        if task is None:
            return
            
        # Get current description without the temporary message
        current_description = task.description
        if "|" in current_description:
            current_description = current_description.split("|")[0].strip()
        
        # Format completed steps if provided
        completed_str = ""
        if completed_steps and completed_steps:
            # Create a numbered list of completed steps
            steps_list = []
            for i, step in enumerate(completed_steps, 1):
                steps_list.append(f"{i}. [green]✓[/green] {step}")
            completed_str = "\n   " + "\n   ".join(steps_list)
            if completed_str:
                completed_str = f"\n[dim]Completed:[/dim]{completed_str}"
        
        # Update with new temporary message and completed steps list
        new_description = f"{current_description} | [yellow]{message}[/yellow]{completed_str}"
        progress.update(task_id, description=new_description)
        
        # Sleep without blocking the display
        time.sleep(duration)
        
        # Restore original description with completed steps
        if completed_steps:
            current_description = f"{current_description}{completed_str}"
        progress.update(task_id, description=current_description)
    except Exception as e:
        # Log error but don't crash
        console.print(f"[dim red]Progress update error: {str(e)}[/dim red]", style="dim")

def clean_html_for_translation(html_content):
    """First layer: Clean HTML and prepare for translation."""
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Remove unnecessary tags
    for tag in soup.find_all(['script', 'style', 'meta', 'link', 'iframe', 'svg']):
        tag.decompose()
    
    # Process specific content tags
    translatable_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'th', 'td', 'div', 'span', 'a']
    
    # Dictionary to store original HTML structure
    html_structure = []
    
    for tag in soup.find_all(translatable_tags):
        if tag.string and tag.string.strip():
            # Store the tag info and its content
            html_structure.append({
                'tag_name': tag.name,
                'attrs': tag.attrs,
                'content': tag.string.strip(),
                'original_html': str(tag)
            })
    
    return html_structure

def restore_html_structure(html_structure, translations):
    """Restore HTML structure with translated content."""
    restored_html = []
    
    for orig, trans in zip(html_structure, translations.split('\n\n')):
        if trans.strip():
            # Create a new tag with original attributes
            soup = BeautifulSoup('', 'lxml')
            new_tag = soup.new_tag(orig['tag_name'], **orig['attrs'])
            new_tag.string = trans.strip()
            restored_html.append(str(new_tag))
    
    return '\n'.join(restored_html)

def extract_text_from_file(file_path):
    """Extract text from various file formats."""
    extension = file_path.suffix.lower()
    
    try:
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif extension == '.html':
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                # First layer: Clean and structure HTML
                html_structure = clean_html_for_translation(html_content)
                # Extract text for translation
                return '\n'.join(item['content'] for item in html_structure), html_structure
        
        elif extension == '.pdf':
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif extension == '.docx':
            doc = Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        elif extension == '.doc':
            console.print("[yellow]Warning: .doc files are not fully supported. Please convert to .docx for better results.[/yellow]")
            return None
        
        elif extension == '.rtf':
            with open(file_path, 'r', encoding='utf-8') as f:
                return rtf_to_text(f.read())
        
        else:
            console.print(f"[red]Unsupported file format: {extension}[/red]")
            return None
            
    except Exception as e:
        console.print(f"[red]Error processing {file_path.name}: {str(e)}[/red]")
        return None

def process_files():
    """Main function to process files with enhanced progress tracking."""
    translation_manager = TranslationManager()
    
    # Get translation direction
    direction = get_translation_direction()
    
    # Setup input and output directories
    input_dir = Path("to_trnslt")
    output_dir = Path("trnsltd")
    output_dir.mkdir(exist_ok=True)
    
    # Get list of files to translate
    files_to_translate = list(input_dir.glob("*"))
    
    if not files_to_translate:
        console.print("[red]No files found in the 'to_trnslt' directory![/red]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        main_task = progress.add_task(
            "[cyan]Processing files...",
            total=len(files_to_translate)
        )
        
        for file_path in files_to_translate:
            file_task = progress.add_task(
                f"[yellow]Processing {file_path.name}...",
                total=100
            )
            
            try:
                # Extract text
                show_temp_status(progress, file_task, "📄 Extracting text...")
                text = extract_text_from_file(file_path)
                
                if text:
                    if isinstance(text, tuple):  # HTML file
                        text_content, html_structure = text
                        translated_text = translation_manager.translate_text(
                            text_content,
                            direction,
                            progress,
                            file_task
                        )
                        final_content = restore_html_structure(html_structure, translated_text)
                    else:  # Other file types
                        translated_text = translation_manager.translate_text(
                            text,
                            direction,
                            progress,
                            file_task
                        )
                        final_content = translated_text
                    
                    # Save translation
                    output_path = output_dir / f"translated_{file_path.name}"
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(final_content)
                    
                    show_temp_status(
                        progress,
                        file_task,
                        f"✅ Successfully translated and saved to {output_path.name}",
                        duration=2
                    )
                
                progress.update(file_task, completed=100)
                progress.update(main_task, advance=1)
                
            except Exception as e:
                console.print(f"[red]Error processing {file_path.name}: {str(e)}[/red]")
                progress.update(main_task, advance=1)
                continue
    
    console.print("[green]Translation process completed![/green]")

if __name__ == "__main__":
    process_files()
