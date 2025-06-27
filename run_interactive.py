#!/usr/bin/env python3
"""
Phoenix Agent Interactive Launcher
Simple launcher script for the Phoenix Agent interactive mode.
Just run this script to start the interactive file selection process.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Launch Phoenix Agent in interactive mode."""
    try:
        # Import and run the orchestrator in interactive mode
        from phoenix_orchestrator import PhoenixOrchestrator
        
        print("🚀 Phoenix Agent Interactive Launcher")
        print("="*50)
        
        # Create orchestrator and run interactive mode
        orchestrator = PhoenixOrchestrator()
        success = orchestrator.interactive_run()
        
        if success:
            print("\n🎉 Translation completed successfully!")
            print("Check your output directory for the translated files.")
        else:
            print("\n❌ Translation failed or was cancelled.")
        
        # Keep console window open on Windows
        if os.name == 'nt':  # Windows
            input("\nPress Enter to exit...")
            
    except ImportError as e:
        print(f"❌ Error importing Phoenix Agent: {e}")
        print("Make sure all required files are in the same directory.")
        if os.name == 'nt':  # Windows
            input("\nPress Enter to exit...")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if os.name == 'nt':  # Windows
            input("\nPress Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main() 