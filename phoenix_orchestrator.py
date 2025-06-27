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

# Import all stations
from document_blueprint import load_blueprint, save_blueprint, validate_blueprint
from document_parser import parse_pdf_to_blueprint
from visual_debugger import VisualDebugger
from translation_service import translate_blueprint
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
        
        # Initialize debugger
        self.debugger = VisualDebugger(str(self.output_dir / "debug_output"))
    
    def run_station_1_surveyor(self, pdf_path: str, page_images_dir: Optional[str] = None) -> bool:
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
            blueprint = parse_pdf_to_blueprint(pdf_path, output_json=str(self.blueprint_path))
            
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
            # Translate blueprint
            success = translate_blueprint(
                str(self.blueprint_path),
                str(self.translated_blueprint_path),
                target_language
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
                         skip_verification: bool = False) -> bool:
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
        if not self.run_station_1_surveyor(pdf_path, page_images_dir):
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


def main():
    """Main entry point for the Phoenix Agent orchestrator."""
    parser = argparse.ArgumentParser(description="Phoenix Agent: Document Translation Pipeline")
    parser.add_argument("pdf_path", help="Path to input PDF file")
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
    
    args = parser.parse_args()
    
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
            success = orchestrator.run_station_1_surveyor(args.pdf_path, args.page_images)
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
            args.skip_verification
        )
    
    if not success:
        print("\n❌ Pipeline failed!")
        sys.exit(1)
    else:
        print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main() 