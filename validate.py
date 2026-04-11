#!/usr/bin/env python3
"""
Validation script for Financial Dashboard implementation.
Checks that all required components are present and functional.
"""
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if Path(filepath).exists():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description} - NOT FOUND")
        return False

def check_module_import(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        print(f"✗ {description} - IMPORT ERROR: {e}")
        return False

def main():
    print("=" * 70)
    print("Financial Dashboard Implementation Validation")
    print("=" * 70)
    print()
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    all_checks = []
    
    # Backend files
    print("Backend Files:")
    all_checks.append(check_file_exists("app.py", "FastAPI application (app.py)"))
    all_checks.append(check_file_exists("config.py", "Configuration module (config.py)"))
    all_checks.append(check_file_exists("model_service.py", "Model service (model_service.py)"))
    all_checks.append(check_file_exists("data_fetcher.py", "Data fetcher (data_fetcher.py)"))
    all_checks.append(check_file_exists("news_service.py", "News service (news_service.py)"))
    print()
    
    # Frontend
    print("Frontend:")
    all_checks.append(check_file_exists("dashboard.html", "Dashboard UI (dashboard.html)"))
    print()
    
    # Tests
    print("Tests:")
    all_checks.append(check_file_exists("tests/__init__.py", "Test package init"))
    all_checks.append(check_file_exists("tests/test_model_service.py", "Model service tests"))
    all_checks.append(check_file_exists("tests/test_news_service.py", "News service tests"))
    all_checks.append(check_file_exists("tests/test_integration.py", "Integration tests"))
    print()
    
    # Configuration & Dependencies
    print("Configuration & Dependencies:")
    all_checks.append(check_file_exists("requirements.txt", "Dependencies (requirements.txt)"))
    all_checks.append(check_file_exists(".gitignore", "Git ignore rules"))
    all_checks.append(check_file_exists("README.md", "Documentation (README.md)"))
    print()
    
    # Models
    print("Pre-trained Models:")
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        print(f"✓ Models directory exists with {len(model_files)} PyTorch model files")
        all_checks.append(True)
    else:
        print("⚠ Models directory not found (needed for inference)")
        all_checks.append(False)
    print()
    
    # Module imports
    print("Python Module Imports:")
    all_checks.append(check_module_import("config", "Config module import"))
    all_checks.append(check_module_import("model_service", "Model service import"))
    all_checks.append(check_module_import("data_fetcher", "Data fetcher import"))
    all_checks.append(check_module_import("news_service", "News service import"))
    all_checks.append(check_module_import("utils", "Neural network utilities"))
    print()
    
    # Summary
    print("=" * 70)
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"Summary: {passed}/{total} checks passed ({percentage:.0f}%)")
    
    if passed == total:
        print("✓ All components verified successfully!")
        return 0
    else:
        print("⚠ Some checks failed - see above for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
