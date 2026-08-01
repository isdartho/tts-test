"""
Requirements helper script for the TTS library.
Verifies installed dependencies and automatically installs missing requirements.
"""

import subprocess
import sys

REQUIRED_PACKAGES = {
    "kokoro_onnx": "kokoro-onnx>=0.4.7",
    "soundfile": "soundfile>=0.14.0",
    "numpy": "numpy>=2.0.0",
    "onnxruntime": "onnxruntime>=1.20.0",
}


def check_and_install():
    print("Checking dependencies for Python TTS library...")
    missing = []

    for mod_name, pkg_spec in REQUIRED_PACKAGES.items():
        try:
            __import__(mod_name)
            print(f"  [✓] {mod_name} is installed.")
        except ImportError:
            print(f"  [✗] {mod_name} is missing.")
            missing.append(pkg_spec)

    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}...")
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        subprocess.check_call(cmd)
        print("\nAll dependencies installed successfully!")
    else:
        print("\nAll required dependencies are satisfied.")


if __name__ == "__main__":
    check_and_install()
