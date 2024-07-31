import sys
import importlib
import argparse

TEXT_DESCRIPTION_WIDTH = 20

def test_python_version_specific(recommended_version, verbose=False):
    found_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    result = "Pass" if found_version == recommended_version else "Fail"
    print(f"{'Test Python Version Specific:':<35} {result}")
    if verbose:
        print(f"\tRecommended Version: {recommended_version}")
        print(f"\tFound version: {found_version}")

def test_python_version_range(allowable_versions, verbose=False):
    allowable_versions = [f"{v.split('.')[0]}.{v.split('.')[1]}" for v in allowable_versions]
    major_minor_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    result = "Pass" if major_minor_version in allowable_versions else "Fail"

    print(f"{'Test Python Version Range:':<35} {result}")
    if verbose:
        print(f"\tAllowable Versions: {allowable_versions}")
        found_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"\tFound version: {found_version}")

REQUIREMENTS = """
jupyterlab
hvsrpy
swprocess
swprepost
numpy
obspy
scipy
pandas
matplotlib
"""

def test_python_packages(required_packages, verbose=False):
    result = "Pass"
    details = []
    for package in required_packages.split("\n"):
        if package == "":
            continue
        try:
            importlib.import_module(package)
        except ModuleNotFoundError:
            result = "Fail"
            details.append(f"\t{package}  Not Installed")
        else:
            details.append(f"\t{package}  Installed")

    print(f"{'Test Pip Requirements:':<35} {result}")
    if verbose:
        print("\n".join(details))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    test_python_version_specific(recommended_version="3.8.10", verbose=args.verbose)
    test_python_version_range(allowable_versions=("3.8", "3.9", "3.10"), verbose=args.verbose)
    test_python_packages(required_packages=REQUIREMENTS, verbose=args.verbose)