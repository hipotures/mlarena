import os
import zipfile
import datetime
import time
import argparse
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Playwright not found. Please install it: pip install playwright")
    exit(1)

# Treść promptu wpisywana po załączeniu pliku
PROMPT_TEXT = "Załącznik zawiera archiwum kodu"

def create_archive():
    source_dirs = ['src', 'conf', 'docs']
    specific_scripts = ['mla.py', 'generate_s6e1_grid.py']
    allowed_extensions = {'.py', '.yaml', '.md', '.txt', '.rst'}
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f'mla-{timestamp}.zip'
    archive_path = os.path.join('/tmp', archive_name)
    
    print(f"Tworzenie archiwum: {archive_path}")
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Archiwizacja pełnych katalogów (src, conf, docs)
        for s_dir in source_dirs:
            if not os.path.exists(s_dir):
                print(f"Ostrzeżenie: Katalog {s_dir} nie istnieje. Pomijam.")
                continue
                
            for root, dirs, files in os.walk(s_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in allowed_extensions:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, file_path)
        
        # Archiwizacja wybranych plików z katalogu scripts
        if os.path.exists('scripts'):
            for script_file in specific_scripts:
                script_path = os.path.join('scripts', script_file)
                if os.path.exists(script_path):
                    zipf.write(script_path, script_path)
                else:
                    print(f"Ostrzeżenie: Plik {script_path} nie istnieje. Pomijam.")
    
    print(f"Sukces! Archiwum zapisane w {archive_path}")
    return archive_path

def upload_to_chatgpt(archive_path, start_new_session=False, custom_prompt=None, press_enter=False):
    CDP_ENDPOINT = "http://localhost:9222"
    TARGET_URL = "https://chatgpt.com/g/g-p-693834dd119c8191a5e726ee05bb10ec-mlarena/project"
    
    print(f"Connecting to CDP at {CDP_ENDPOINT}...")
    
    with sync_playwright() as p:
        try:
            # Connect to existing browser session via CDP
            browser = p.chromium.connect_over_cdp(CDP_ENDPOINT)
            
            # Use the first available context
            if not browser.contexts:
                context = browser.new_context()
            else:
                context = browser.contexts[0]
            
            # Use the first page or create new one if none exists
            if not context.pages:
                page = context.new_page()
            else:
                page = context.pages[0]
            
            if start_new_session:
                print(f"Navigating to {TARGET_URL}...")
                page.goto(TARGET_URL)
                page.wait_for_load_state("domcontentloaded")
            else:
                print("Using current page context (assuming ChatGPT is open)...")
                try:
                    page.bring_to_front()
                except:
                    pass
            
            # Wait for page to be interactable
            try:
                page.wait_for_selector('#prompt-textarea', timeout=10000)
            except Exception:
                print("Warning: Prompt textarea not found immediately.")

            print(f"Uploading file: {archive_path}")
            
            try:
                # 1. Click the "Add files" button (plus icon)
                add_button = page.locator('button[aria-label="Add files and more"]').first
                if add_button.is_visible():
                    add_button.click()
                    time.sleep(1)
                else:
                    print("Specific 'Add files and more' button not found, searching broadly...")
                    add_button = page.locator('button[aria-label*="Add files"]').first
                    if add_button.is_visible():
                        add_button.click()
                        time.sleep(1)

                # 2. Handle the file chooser event
                with page.expect_file_chooser(timeout=5000) as fc_info:
                    menu_item = page.get_by_role("menuitem", name="Add photos & files")
                    if menu_item.count() > 0 and menu_item.is_visible():
                        menu_item.click()
                    else:
                        print("Menu item not found, delegating to fallback...")
                        raise Exception("Menu item 'Add photos & files' not visible/found")

                file_chooser = fc_info.value
                file_chooser.set_files(archive_path)
                print("File uploaded via FileChooser!")

            except Exception as e:
                print(f"Standard UI interaction failed ({e}), trying direct input manipulation...")
                file_input = page.locator('input[type="file"]').first
                if file_input.count() > 0:
                    try:
                        page.evaluate("document.querySelector('input[type=\"file\"]').style.display = 'block'")
                    except:
                        pass
                    file_input.set_input_files(archive_path)
                    print("File set to input via fallback.")
                else:
                    print("Error: Could not find file input element.")

            time.sleep(3)
            
            print("Typing prompt...")
            prompt_selector = '#prompt-textarea'
            
            # Use provided prompt or default
            text_to_type = custom_prompt if custom_prompt else PROMPT_TEXT
            page.fill(prompt_selector, text_to_type)
            
            if press_enter:
                # Press Enter to send
                print("Sending message (Enter)...")
                page.press(prompt_selector, "Enter")
                print("Done! Message sent.")
            else:
                # Move to new line using Shift+Enter (default)
                print("Adding newline (Shift+Enter)...")
                page.press(prompt_selector, "Shift+Enter")
                print("Done! File uploaded and prompt filled. You can now press send manually.")
            
        except Exception as e:
            print(f"Error during CDP interaction: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Archive project and upload to ChatGPT via CDP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples of usage:
  Standard (default prompt, adds newline):
    python scripts/upload_to_mlarena.py

  With custom prompt:
    python scripts/upload_to_mlarena.py --prompt "Analyze this code structure"

  Send immediately (Presses Enter):
    python scripts/upload_to_mlarena.py --enter

  Start new session (Navigate to URL) and send:
    python scripts/upload_to_mlarena.py --start --prompt "Project update" --enter
        """
    )
    parser.add_argument("--start", action="store_true", help="Navigate to the project URL (start new session). Default is to use current page.")
    parser.add_argument("--prompt", type=str, help="Custom prompt text to type. Overrides default.")
    parser.add_argument("--enter", action="store_true", help="Press Enter to send the message immediately instead of Shift+Enter (newline).")
    args = parser.parse_args()

    archive = create_archive()
    upload_to_chatgpt(archive, start_new_session=args.start, custom_prompt=args.prompt, press_enter=args.enter)