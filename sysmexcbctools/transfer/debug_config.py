"""
Debug script to check configuration and file access.
Run this to diagnose any issues with your configuration setup.
"""

import sys
from pathlib import Path

import numpy as np

# Add current directory to Python path
sys.path.insert(0, '.')

try:
    from config.config_loader import ConfigLoader
    from utils.data_loader import DataLoader
    print("✓ Successfully imported configuration modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    print("Make sure you have the config and utils directories with the Python files")
    sys.exit(1)


def check_config_file():
    """Check if the YAML config file exists and can be loaded."""
    config_file = Path("config/data_paths.yaml")

    if not config_file.exists():
        print(f"✗ Configuration file not found: {config_file}")
        return False

    print(f"✓ Configuration file exists: {config_file}")

    try:
        loader = ConfigLoader()
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False


def check_file_paths():
    """Check specific file paths from your configuration."""

    try:
        loader = ConfigLoader()

        # Check the strides centile samples file specifically
        print("\nChecking strides centile samples file:")
        try:
            file_path = loader.get_file_path('centile_samples', 'strides')
            print(f"  Config path: {file_path}")
            print(f"  Path exists: {file_path.exists()}")

            if file_path.exists():
                print(f"  File size: {file_path.stat().st_size} bytes")

                # Try to get basic info about the numpy file
                try:
                    # First try without allow_pickle
                    with np.load(file_path, allow_pickle=False) as data:
                        print("  ✓ File loads without pickle")
                except ValueError as e:
                    if "allow_pickle" in str(e):
                        print("  ! File requires pickle to load")
                        try:
                            data = np.load(file_path, allow_pickle=True)
                            print("  ✓ File loads with allow_pickle=True")
                            print(f"  ✓ Data shape: {data.shape}")
                            print(f"  ✓ Data dtype: {data.dtype}")
                        except Exception as e2:
                            print(f"  ✗ Failed to load with pickle: {e2}")
                    else:
                        print(f"  ✗ Other numpy error: {e}")
            else:
                print("  ✗ File does not exist")

        except Exception as e:
            print(f"✗ Error checking strides file: {e}")

    except Exception as e:
        print(f"✗ Error creating config loader: {e}")


def check_all_configured_paths():
    """Check all paths in the configuration."""

    try:
        loader = ConfigLoader()
        print("\nValidating all configured paths:")

        validation_results = loader.validate_all_paths()

        existing_paths = []
        missing_paths = []

        for path_name, exists in validation_results.items():
            if exists:
                existing_paths.append(path_name)
                print(f"  ✓ {path_name}")
            else:
                missing_paths.append(path_name)
                print(f"  ✗ {path_name}")

        print(f"\nSummary: {len(existing_paths)} paths exist, {len(missing_paths)} missing")

        if missing_paths:
            print("\nMissing paths:")
            for path in missing_paths:
                try:
                    if 'files.centile_samples' in path:
                        file_path = loader.get_file_path('centile_samples', path.split('.')[-1])
                    elif 'datasets.' in path:
                        parts = path.split('.')
                        category, dataset = parts[1], parts[2]
                        file_path = loader.get_dataset_path(category, dataset)
                    else:
                        continue
                    print(f"  Expected at: {file_path}")
                except:
                    pass

    except Exception as e:
        print(f"✗ Error validating paths: {e}")


def test_data_loading():
    """Test loading data using the new system."""

    print("\nTesting data loading:")

    try:
        from utils.data_loader import load_centile_samples

        print("Testing strides centile samples:")
        try:
            data = load_centile_samples('strides')
            print("  ✓ Successfully loaded strides centile samples")
            print(f"  ✓ Shape: {data.shape}")
            print(f"  ✓ Dtype: {data.dtype}")

        except FileNotFoundError as e:
            print(f"  ✗ File not found: {e}")
        except ValueError as e:
            print(f"  ✗ Value error (pickle issue): {e}")
        except Exception as e:
            print(f"  ✗ Other error: {e}")

    except Exception as e:
        print(f"✗ Error importing data loader: {e}")


def check_symlink_removal():
    """Check if symlinks were properly removed."""

    print("\nChecking for remaining symlinks:")

    # Look for any remaining symlinks
    import subprocess
    try:
        result = subprocess.run(['find', '.', '-maxdepth', '3', '-type', 'l'],
                              capture_output=True, text=True)

        if result.returncode == 0:
            remaining_symlinks = result.stdout.strip().split('\n') if result.stdout.strip() else []

            if remaining_symlinks and remaining_symlinks[0]:  # Check if there are actual symlinks
                print(f"  ! Found {len(remaining_symlinks)} remaining symlinks:")
                for symlink in remaining_symlinks:
                    print(f"    - {symlink}")
            else:
                print("  ✓ No remaining symlinks found")
        else:
            print(f"  ? Could not check for symlinks: {result.stderr}")

    except Exception as e:
        print(f"  ? Error checking symlinks: {e}")


def main():
    """Run all diagnostic checks."""

    print("CONFIGURATION DIAGNOSTIC")
    print("=" * 60)

    # Check basic setup
    if not check_config_file():
        return

    # Check file paths
    check_file_paths()

    # Check all configured paths
    check_all_configured_paths()

    # Test data loading
    test_data_loading()

    # Check symlink removal
    check_symlink_removal()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")

    print("\nIf you see errors above:")
    print("1. Check that all file paths in config/data_paths.yaml are correct")
    print("2. Verify you have access to the RDS filesystem")
    print("3. Make sure the NumPy files aren't corrupted")
    print("4. Try loading files manually with allow_pickle=True")


if __name__ == "__main__":
    main()
