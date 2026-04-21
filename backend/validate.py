#!/usr/bin/env python3
"""
Validation script for Financial Dashboard implementation.
Checks that all required components are present and functional.
Aligned with backend-only module architecture.
"""
import os
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if Path(filepath).exists():
        print(f"[OK] {description}")
        return True
    print(f"[FAIL] {description} - NOT FOUND")
    return False


def check_module_import(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"[OK] {description}")
        return True
    except ImportError as e:
        print(f"[FAIL] {description} - IMPORT ERROR: {e}")
        return False


def main():
    print("=" * 70)
    print("Financial Dashboard Implementation Validation")
    print("=" * 70)
    print()

    # Change to project root directory
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    all_checks = []

    # Backend files
    print("Backend Files:")
    backend_files = [
        "backend/__init__.py",
        "backend/app.py",
        "backend/config.py",
        "backend/model_service.py",
        "backend/data_fetcher.py",
        "backend/news_service.py",
        "backend/inference_service.py",
        "backend/options_analyzer.py",
        "backend/utils.py",
        "backend/utils_2.py",
        "backend/validate.py",
    ]
    for backend_file in backend_files:
        all_checks.append(check_file_exists(backend_file, f"Backend file ({backend_file})"))
    print()

    # Frontend
    print("Frontend:")
    all_checks.append(check_file_exists("frontend/templates/dashboard.html", "Dashboard template"))
    all_checks.append(check_file_exists("frontend/static/css/dashboard.css", "Dashboard stylesheet"))
    print()

    # Tests
    print("Tests:")
    all_checks.append(check_file_exists("tests/__init__.py", "Test package init"))
    all_checks.append(check_file_exists("tests/test_model_service.py", "Model service tests"))
    all_checks.append(check_file_exists("tests/test_inference_service.py", "Inference service tests"))
    all_checks.append(check_file_exists("tests/test_news_service.py", "News service tests"))
    all_checks.append(check_file_exists("tests/test_integration.py", "Integration tests"))
    all_checks.append(check_file_exists("tests/test_dashboard_html.py", "Dashboard HTML tests"))
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
        print(f"[OK] Models directory exists with {len(model_files)} PyTorch model files")
        all_checks.append(True)
    else:
        print("[FAIL] Models directory not found (needed for inference)")
        all_checks.append(False)
    print()

    # Module imports
    print("Python Module Imports:")
    all_checks.append(check_module_import("backend.config", "Config module import"))
    all_checks.append(check_module_import("backend.model_service", "Model service import"))
    all_checks.append(check_module_import("backend.data_fetcher", "Data fetcher import"))
    all_checks.append(check_module_import("backend.news_service", "News service import"))
    all_checks.append(check_module_import("backend.inference_service", "Inference service import"))
    all_checks.append(check_module_import("backend.options_analyzer", "Options analyzer import"))
    all_checks.append(check_module_import("backend.utils", "Neural network utilities"))
    all_checks.append(check_module_import("backend.utils_2", "Feature engineering utilities"))
    print()

    # Summary
    print("=" * 70)
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total * 100) if total > 0 else 0

    print(f"Summary: {passed}/{total} checks passed ({percentage:.0f}%)")

    if passed == total:
        print("[OK] All components verified successfully!")
        return 0
    print("[WARN] Some checks failed - see above for details")
    return 1


if __name__ == "__main__":
    sys.exit(main())
