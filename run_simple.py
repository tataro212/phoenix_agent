#!/usr/bin/env python3
"""
Simple Phoenix Agent Runner
Alternative to interactive mode that uses command line input instead of tkinter dialogs.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phoenix_orchestrator import PhoenixOrchestrator

def main():
    """Simple interactive mode using command line input."""
    print("🎯 PHOENIX AGENT - SIMPLE INTERACTIVE MODE")
    print("="*50)
    print("This mode uses command line input for file selection.")
    print("="*50)
    
    # Create orchestrator
    orchestrator = PhoenixOrchestrator()
    
    # Get input file
    print("\n📁 Step 1: Enter input PDF file path")
    print("(You can drag and drop the PDF file here, or type the path)")
    pdf_path = input("PDF file path: ").strip().strip('"')  # Remove quotes if dragged
    
    if not pdf_path:
        print("❌ No file path provided. Exiting.")
        return False
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return False
    
    # Get output directory
    print("\n📁 Step 2: Enter output directory path")
    print("(Press Enter to use default: ./phoenix_output)")
    output_dir = input("Output directory: ").strip().strip('"')
    
    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "phoenix_output")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ Created output directory: {output_dir}")
        except Exception as e:
            print(f"❌ Error creating directory: {e}")
            return False
    
    # Update orchestrator output directory
    orchestrator.output_dir = Path(output_dir)
    orchestrator.output_dir.mkdir(exist_ok=True)
    
    # Update file paths for new output directory
    orchestrator.blueprint_path = orchestrator.output_dir / "document_blueprint.json"
    orchestrator.translated_blueprint_path = orchestrator.output_dir / "translated_blueprint.json"
    orchestrator.reconstructed_pdf_path = orchestrator.output_dir / "reconstructed_document.pdf"
    orchestrator.toc_pdf_path = orchestrator.output_dir / "table_of_contents.pdf"
    orchestrator.element_page_map_path = orchestrator.output_dir / "element_page_map.json"
    
    # Set target language to Greek
    target_language = 'el'
    print(f"\n🌍 Step 3: Target language is set to Greek (el)")
    
    # Confirm settings
    print("\n" + "="*50)
    print("📋 CONFIRMATION")
    print("="*50)
    print(f"Input PDF: {pdf_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Language: {target_language}")
    print("="*50)
    
    # Ask for confirmation
    response = input("\nProceed with translation? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("❌ Translation cancelled.")
        return False
    
    # Run the pipeline
    print("\n🚀 Starting translation pipeline...")
    return orchestrator.run_full_pipeline(pdf_path, target_language)

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Pipeline failed!")
        sys.exit(1)
    else:
        print("\n✅ Pipeline completed successfully!") 