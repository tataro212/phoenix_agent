#!/usr/bin/env python3
"""
Command-line version of Phoenix Agent that bypasses interactive mode
"""

import sys
import os
from pathlib import Path
from phoenix_orchestrator import PhoenixOrchestrator

def main():
    """Run Phoenix Agent in command-line mode"""
    print("🚀 Phoenix Agent - Command Line Mode")
    print("="*50)
    
    # Check if PDF file is provided
    if len(sys.argv) < 2:
        print("Usage: python run_command_line.py <input.pdf> [output_directory]")
        print("Example: python run_command_line.py document.pdf ./output")
        return False
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "phoenix_output"
    
    # Check if PDF exists
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"📄 Input PDF: {pdf_path}")
    print(f"📁 Output Directory: {output_path}")
    print(f"🌍 Target Language: Greek (el)")
    print("="*50)
    
    # Create orchestrator
    orchestrator = PhoenixOrchestrator(str(output_path))
    
    # Update file paths for the new output directory
    orchestrator.blueprint_path = output_path / "document_blueprint.json"
    orchestrator.translated_blueprint_path = output_path / "translated_blueprint.json"
    orchestrator.reconstructed_pdf_path = output_path / "reconstructed_document.pdf"
    orchestrator.toc_pdf_path = output_path / "table_of_contents.pdf"
    orchestrator.element_page_map_path = output_path / "element_page_map.json"
    
    # Set target language to Greek
    target_language = 'el'
    
    print("🚀 Starting translation pipeline...")
    
    # Run the pipeline
    success = orchestrator.run_full_pipeline(pdf_path, target_language)
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        print(f"📄 Reconstructed Document: {orchestrator.reconstructed_pdf_path}")
        print(f"📋 Table of Contents: {orchestrator.toc_pdf_path}")
        return True
    else:
        print("\n❌ Pipeline failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 