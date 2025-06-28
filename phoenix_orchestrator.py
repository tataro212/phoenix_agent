"""
Phoenix Agent Orchestrator
Coordinates all four stations of the Phoenix Agent system following the assembly line approach.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import time

# Import tkinter for file dialogs
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Import all stations
from document_blueprint import load_blueprint, save_blueprint, validate_blueprint
from document_parser import parse_pdf_to_blueprint
from visual_debugger import VisualDebugger
from translation_service import translate_blueprint, GeminiAPI, DummyGeminiAPI
from document_reconstructor import reconstruct_from_blueprint
from toc_generator import create_toc_from_reconstruction


class PhoenixOrchestrator:
    """Main orchestrator for the Phoenix Agent system."""
    
    def __init__(self, output_dir: str = "phoenix_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # File paths for intermediate outputs
        self.blueprint_path = self.output_dir / "document_blueprint.json"
        self.translated_blueprint_path = self.output_dir / "translated_blueprint.json"
        self.reconstructed_pdf_path = self.output_dir / "reconstructed_document.pdf"
        self.toc_pdf_path = self.output_dir / "table_of_contents.pdf"
        self.element_page_map_path = self.output_dir / "element_page_map.json"
        
        # Initialize translation API
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            # Try to read from GEMINI_API_KEY.txt in the project root
            key_file = Path(__file__).parent / "GEMINI_API_KEY.txt"
            if key_file.exists():
                with open(key_file, "r") as f:
                    self.gemini_api_key = f.read().strip()
            if not self.gemini_api_key:
                print("⚠️  WARNING: GEMINI_API_KEY environment variable not set and GEMINI_API_KEY.txt not found. Using DUMMY translator.")
                self.translator_client = DummyGeminiAPI()
            else:
                try:
                    self.translator_client = GeminiAPI(self.gemini_api_key)
                except Exception as e:
                    print(f"⚠️  Failed to initialize real Gemini API: {e}")
                    print("⚠️  Falling back to DUMMY translator.")
                    self.translator_client = DummyGeminiAPI()
        else:
            try:
                self.translator_client = GeminiAPI(self.gemini_api_key)
            except Exception as e:
                print(f"⚠️  Failed to initialize real Gemini API: {e}")
                print("⚠️  Falling back to DUMMY translator.")
                self.translator_client = DummyGeminiAPI()
        
        # Initialize debugger
        self.debugger = VisualDebugger(str(self.output_dir / "debug_output"))
    
    def select_input_file(self) -> Optional[str]:
        """Open file dialog to select input PDF file."""
        if not TKINTER_AVAILABLE:
            print("❌ Error: tkinter not available. Please install tkinter or use command line arguments.")
            return None
        
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Force the dialog to appear on top
        root.attributes('-topmost', True)
        root.focus_force()
        root.lift()
        root.update()
        time.sleep(0.1)  # Small delay to ensure dialog appears
        
        try:
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select PDF file to translate",
                filetypes=[
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ],
                initialdir=os.getcwd(),
                parent=root
            )
            
            if file_path:
                print(f"✅ Selected input file: {file_path}")
                return file_path
            else:
                print("❌ No file selected")
                return None
                
        except Exception as e:
            print(f"❌ Error selecting file: {e}")
            return None
        finally:
            root.destroy()
    
    def select_output_directory(self) -> Optional[str]:
        """Open folder dialog to select output directory."""
        if not TKINTER_AVAILABLE:
            print("❌ Error: tkinter not available. Please install tkinter or use command line arguments.")
            return None
        
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Force the dialog to appear on top
        root.attributes('-topmost', True)
        root.focus_force()
        root.lift()
        root.update()
        time.sleep(0.1)  # Small delay to ensure dialog appears
        
        try:
            # Open folder dialog
            folder_path = filedialog.askdirectory(
                title="Select folder to save translation output",
                initialdir=os.getcwd(),
                parent=root
            )
            
            if folder_path:
                print(f"✅ Selected output directory: {folder_path}")
                return folder_path
            else:
                print("❌ No directory selected")
                return None
                
        except Exception as e:
            print(f"❌ Error selecting directory: {e}")
            return None
        finally:
            root.destroy()
    
    def select_target_language(self) -> str:
        """Show language selection dialog."""
        if not TKINTER_AVAILABLE:
            print("❌ Error: tkinter not available. Using default language 'en'")
            return "en"
        
        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Common languages
        languages = {
            "English": "en",
            "Spanish": "es", 
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Russian": "ru",
            "Chinese (Simplified)": "zh",
            "Japanese": "ja",
            "Korean": "ko",
            "Arabic": "ar",
            "Hindi": "hi"
        }
        
        try:
            # Create a simple dialog
            dialog = tk.Toplevel(root)
            dialog.title("Select Target Language")
            dialog.geometry("300x400")
            dialog.resizable(False, False)
            
            # Center the dialog
            dialog.transient(root)
            dialog.grab_set()
            
            # Add label
            label = tk.Label(dialog, text="Select target language for translation:", pady=10)
            label.pack()
            
            selected_language = tk.StringVar(value="en")
            
            # Create radio buttons for each language
            for lang_name, lang_code in languages.items():
                rb = tk.Radiobutton(
                    dialog, 
                    text=lang_name, 
                    variable=selected_language, 
                    value=lang_code
                )
                rb.pack(anchor=tk.W, padx=20, pady=2)
            
            # Add OK button
            def on_ok():
                dialog.destroy()
            
            ok_button = tk.Button(dialog, text="OK", command=on_ok, width=10)
            ok_button.pack(pady=20)
            
            # Wait for dialog to close
            dialog.wait_window()
            
            selected = selected_language.get()
            print(f"✅ Selected target language: {selected}")
            return selected
            
        except Exception as e:
            print(f"❌ Error selecting language: {e}")
            return "en"
        finally:
            root.destroy()
    
    def run_station_1_surveyor(self, pdf_path: str, page_images_dir: Optional[str] = None, nougat_model_path: Optional[str] = None) -> bool:
        """
        Station 1: The Surveyor
        Parses PDF to create document_blueprint.json
        """
        print("\n" + "="*60)
        print("STATION 1: THE SURVEYOR")
        print("="*60)
        print(f"Input: {pdf_path}")
        print(f"Output: {self.blueprint_path}")
        
        try:
            # Parse PDF to blueprint
            blueprint = parse_pdf_to_blueprint(pdf_path, output_json=str(self.blueprint_path), nougat_model_path=nougat_model_path)
            
            if not blueprint:
                print("❌ Station 1 failed: Could not create blueprint")
                return False
            
            # Validate blueprint
            if not validate_blueprint(blueprint):
                print("❌ Station 1 failed: Invalid blueprint structure")
                return False
            
            print("✅ Station 1 completed: Blueprint created and validated")
            return True
            
        except Exception as e:
            print(f"❌ Station 1 failed: {e}")
            return False
    
    def run_station_1_verification(self, page_images_dir: Optional[str] = None) -> bool:
        """
        Visual verification of Station 1 output
        """
        print("\n" + "-"*40)
        print("STATION 1 VERIFICATION")
        print("-"*40)
        
        try:
            # Load blueprint for verification
            blueprint = load_blueprint(str(self.blueprint_path))
            if not blueprint:
                print("❌ Verification failed: Could not load blueprint")
                return False
            
            # Run visual debugger
            if page_images_dir and Path(page_images_dir).exists():
                success = self.debugger.debug_blueprint(blueprint, page_images_dir)
                if success:
                    print("✅ Visual verification completed")
                    print(f"   Debug images saved to: {self.output_dir / 'debug_output'}")
                    return True
                else:
                    print("❌ Visual verification failed")
                    return False
            else:
                print("⚠️  Skipping visual verification (no page images provided)")
                return True
                
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def run_station_2_diplomat(self, target_language: str = "en") -> bool:
        """
        Station 2: The Diplomat
        Translates blueprint content with context-aware chunking
        """
        print("\n" + "="*60)
        print("STATION 2: THE DIPLOMAT")
        print("="*60)
        print(f"Input: {self.blueprint_path}")
        print(f"Output: {self.translated_blueprint_path}")
        print(f"Target Language: {target_language}")
        
        try:
            # Translate blueprint with strict JSON contract enforcement
            success = translate_blueprint(
                str(self.blueprint_path),
                str(self.translated_blueprint_path),
                target_language,
                gemini_api=self.translator_client,  # Pass the initialized client
                strict_json_contract=True  # Enforce strict JSON contract
            )
            
            if success:
                print("✅ Station 2 completed: Blueprint translated")
                return True
            else:
                print("❌ Station 2 failed: Translation failed")
                return False
                
        except Exception as e:
            print(f"❌ Station 2 failed: {e}")
            return False
    
    def run_station_3_architect(self) -> bool:
        """
        Station 3: The Architect
        Reconstructs document from translated blueprint
        """
        print("\n" + "="*60)
        print("STATION 3: THE ARCHITECT")
        print("="*60)
        print(f"Input: {self.translated_blueprint_path}")
        print(f"Output: {self.reconstructed_pdf_path}")
        
        try:
            # Reconstruct document
            element_page_map = reconstruct_from_blueprint(
                str(self.translated_blueprint_path),
                str(self.reconstructed_pdf_path)
            )
            
            # Save element page map for TOC generation
            with open(self.element_page_map_path, 'w') as f:
                json.dump(element_page_map, f, indent=2)
            
            print("✅ Station 3 completed: Document reconstructed")
            print(f"   Element page map saved to: {self.element_page_map_path}")
            return True
            
        except Exception as e:
            print(f"❌ Station 3 failed: {e}")
            return False
    
    def run_station_4_librarian(self) -> bool:
        """
        Station 4: The Librarian
        Generates interactive Table of Contents
        """
        print("\n" + "="*60)
        print("STATION 4: THE LIBRARIAN")
        print("="*60)
        print(f"Input: {self.translated_blueprint_path}")
        print(f"Output: {self.toc_pdf_path}")
        
        try:
            # Generate TOC
            success = create_toc_from_reconstruction(
                str(self.translated_blueprint_path),
                str(self.reconstructed_pdf_path),
                str(self.toc_pdf_path)
            )
            
            if success:
                print("✅ Station 4 completed: Table of Contents generated")
                return True
            else:
                print("❌ Station 4 failed: TOC generation failed")
                return False
                
        except Exception as e:
            print(f"❌ Station 4 failed: {e}")
            return False
    
    def run_full_pipeline(self, pdf_path: str, target_language: str = "en", 
                         page_images_dir: Optional[str] = None, 
                         skip_verification: bool = False,
                         nougat_model_path: Optional[str] = None) -> bool:
        """
        Runs the complete Phoenix Agent pipeline.
        """
        print("🚀 PHOENIX AGENT PIPELINE STARTING")
        print("="*80)
        print(f"Input PDF: {pdf_path}")
        print(f"Target Language: {target_language}")
        print(f"Output Directory: {self.output_dir}")
        print("="*80)
        
        # Station 1: Surveyor
        if not self.run_station_1_surveyor(pdf_path, page_images_dir, nougat_model_path):
            return False
        
        # Station 1 Verification (unless skipped)
        if not skip_verification:
            if not self.run_station_1_verification(page_images_dir):
                print("⚠️  Verification failed, but continuing...")
        
        # Station 2: Diplomat
        if not self.run_station_2_diplomat(target_language):
            return False
        
        # Station 3: Architect
        if not self.run_station_3_architect():
            return False
        
        # Station 4: Librarian
        if not self.run_station_4_librarian():
            return False
        
        print("\n" + "="*80)
        print("🎉 PHOENIX AGENT PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📄 Reconstructed Document: {self.reconstructed_pdf_path}")
        print(f"📋 Table of Contents: {self.toc_pdf_path}")
        print(f"🔍 Debug Output: {self.output_dir / 'debug_output'}")
        print("="*80)
        
        return True

    def interactive_run(self) -> bool:
        """
        Interactive mode using file dialogs for file selection.
        """
        print("🎯 PHOENIX AGENT - INTERACTIVE MODE")
        print("="*50)
        print("This mode will open file dialogs to help you select:")
        print("1. Input PDF file to translate")
        print("2. Output directory for results")
        print("3. Target language for translation (default: Greek)")
        print("="*50)
        
        # Select input file
        print("\n📁 Step 1: Select input PDF file...")
        pdf_path = self.select_input_file()
        if not pdf_path:
            print("❌ No input file selected. Exiting.")
            return False
        
        # Select output directory
        print("\n📁 Step 2: Select output directory...")
        output_dir = self.select_output_directory()
        if not output_dir:
            print("❌ No output directory selected. Exiting.")
            return False
        
        # Update output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Update file paths for new output directory
        self.blueprint_path = self.output_dir / "document_blueprint.json"
        self.translated_blueprint_path = self.output_dir / "translated_blueprint.json"
        self.reconstructed_pdf_path = self.output_dir / "reconstructed_document.pdf"
        self.toc_pdf_path = self.output_dir / "table_of_contents.pdf"
        self.element_page_map_path = self.output_dir / "element_page_map.json"
        
        # Always use Greek as the target language
        target_language = 'el'
        print("\n🌍 Step 3: Target language is set to Greek (el)")
        
        # Confirm settings
        print("\n" + "="*50)
        print("📋 CONFIRMATION")
        print("="*50)
        print(f"Input PDF: {pdf_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Target Language: {target_language}")
        print("="*50)
        
        # Ask for confirmation
        if not TKINTER_AVAILABLE:
            response = input("\nProceed with translation? (y/n): ").lower().strip()
            if response not in ['y', 'yes']:
                print("❌ Translation cancelled.")
                return False
        else:
            # Create confirmation dialog
            root = tk.Tk()
            root.withdraw()
            
            result = messagebox.askyesno(
                "Confirm Translation",
                f"Proceed with translation?\n\nInput: {pdf_path}\nOutput: {self.output_dir}\nLanguage: {target_language}"
            )
            root.destroy()
            
            if not result:
                print("❌ Translation cancelled.")
                return False
        
        # Run the pipeline
        print("\n🚀 Starting translation pipeline...")
        return self.run_full_pipeline(pdf_path, target_language)


def main():
    """Main entry point for the Phoenix Agent orchestrator."""
    parser = argparse.ArgumentParser(description="Phoenix Agent: Document Translation Pipeline")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode with file dialogs")
    parser.add_argument("pdf_path", nargs="?", help="Path to input PDF file (not needed in interactive mode)")
    parser.add_argument("--target-language", "-t", default="en", 
                       help="Target language for translation (default: en)")
    parser.add_argument("--output-dir", "-o", default="phoenix_output",
                       help="Output directory (default: phoenix_output)")
    parser.add_argument("--page-images", "-p", 
                       help="Directory containing page images for verification")
    parser.add_argument("--skip-verification", "-s", action="store_true",
                       help="Skip visual verification step")
    parser.add_argument("--station", choices=["1", "2", "3", "4"],
                       help="Run only a specific station")
    parser.add_argument("--nougat-model-path", default="facebook/nougat-base", help="Path or name of the Nougat model to use (default: facebook/nougat-base)")
    
    args = parser.parse_args()
    
    # Check for interactive mode
    if args.interactive:
        if not TKINTER_AVAILABLE:
            print("❌ Error: Interactive mode requires tkinter. Please install tkinter or use command line arguments.")
            sys.exit(1)
        
        # Create orchestrator with default output dir (will be updated in interactive mode)
        orchestrator = PhoenixOrchestrator()
        
        # Run interactive mode
        success = orchestrator.interactive_run()
        
        if not success:
            print("\n❌ Interactive pipeline failed!")
            sys.exit(1)
        else:
            print("\n✅ Interactive pipeline completed successfully!")
        return
    
    # Command line mode
    if not args.pdf_path:
        print("❌ Error: Input PDF path is required in command line mode.")
        print("Use --interactive or -i for interactive mode with file dialogs.")
        sys.exit(1)
    
    # Validate input file
    if not Path(args.pdf_path).exists():
        print(f"❌ Error: Input PDF not found: {args.pdf_path}")
        sys.exit(1)
    
    # Create orchestrator
    orchestrator = PhoenixOrchestrator(args.output_dir)
    
    # Run pipeline
    if args.station:
        # Run specific station
        station_num = int(args.station)
        if station_num == 1:
            success = orchestrator.run_station_1_surveyor(args.pdf_path, args.page_images, args.nougat_model_path)
        elif station_num == 2:
            success = orchestrator.run_station_2_diplomat(args.target_language)
        elif station_num == 3:
            success = orchestrator.run_station_3_architect()
        elif station_num == 4:
            success = orchestrator.run_station_4_librarian()
    else:
        # Run full pipeline
        success = orchestrator.run_full_pipeline(
            args.pdf_path, 
            args.target_language, 
            args.page_images, 
            args.skip_verification,
            args.nougat_model_path
        )
    
    if not success:
        print("\n❌ Pipeline failed!")
        sys.exit(1)
    else:
        print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main() 