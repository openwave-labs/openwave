"""
OpenWave CLI - Command Line Interface for running Xperiments.

This module provides the command-line entry point for OpenWave,
allowing users to interactively select and run Xperiments.
"""

import argparse
import os
import sys
import subprocess
import webbrowser
from pathlib import Path

# Conditional import for simple_term_menu (not available on Windows)
try:
    from simple_term_menu import TerminalMenu

    HAS_INTERACTIVE_MENU = True
except (ImportError, NotImplementedError):
    HAS_INTERACTIVE_MENU = False

# Hardcoded welcome entry
WELCOME_URL = "https://github.com/openwave-labs/openwave/blob/main/WELCOME.md"
WELCOME_ENTRY = ("README FIRST: WELCOME TO OPENWAVE XPERIMENTS", WELCOME_URL)


def get_experiments_list():
    """
    Get a list of available Xperiment launchers from the xperiments directory.
    Each collection has exactly one _launcher.py file.

    Returns:
        list: List of tuples containing (display_name, file_path)
    """
    # Get the xperiments directory path
    # Navigate from _io module to parent package, then to xperiments
    package_dir = Path(__file__).parent.parent
    xperiments_dir = package_dir / "xperiments"

    if not xperiments_dir.exists():
        print(f"Error: Xperiments directory not found at {xperiments_dir}")
        sys.exit(1)

    experiments = []

    # Find all _launcher.py files in collection subdirectories
    for launcher_path in xperiments_dir.rglob("_launcher.py"):
        # Skip files in directories that start with underscore
        if any(part.startswith("_") for part in launcher_path.parent.parts):
            continue

        # Get collection directory
        collection_dir = launcher_path.parent
        collection_name = collection_dir.name

        # Skip if directly in xperiments root (no collection)
        if collection_dir == xperiments_dir:
            continue

        # Get display name from collection's __init__.py docstring
        display_name = None
        init_path = collection_dir / "__init__.py"

        if init_path.exists():
            try:
                with open(init_path, "r") as f:
                    lines = f.readlines()
                    # Look for first non-empty line in docstring
                    if len(lines) > 0 and '"""' in lines[0]:
                        for line in lines[1:]:
                            if '"""' in line:
                                break
                            stripped = line.strip()
                            if stripped:
                                display_name = stripped
                                break
            except Exception:
                pass

        # Fallback to formatted collection name
        if not display_name:
            display_name = (
                collection_name.replace("__", ": ").replace("_", " ").replace("-", " ").title()
            )

        experiments.append((display_name, str(launcher_path)))

    # Sort by display name (which starts with A/, B/, C/ etc.)
    experiments.sort(key=lambda x: x[0])

    # Insert hardcoded welcome entry at the beginning
    experiments.insert(0, WELCOME_ENTRY)

    return experiments


def show_menu_simple(experiments):
    """
    Display a simple numbered menu for Xperiment selection.
    Fallback for systems where interactive menu is not available.

    Args:
        experiments: List of tuples containing (display_name, file_path)

    Returns:
        tuple: (display_name, file_path) of the selected xperiment
    """
    # Get version from source (works with editable installs)
    try:
        from openwave import __version__

        pkg_version = __version__
    except ImportError:
        # Fallback to metadata if __version__ not available
        from importlib.metadata import version

        pkg_version = version("OPENWAVE")

    print("\n" + "=" * 64)
    print(f"OPENWAVE (v{pkg_version}) - Available XPERIMENTS")
    print("=" * 64 + "\n")

    # Display numbered list of experiments
    for idx, (display_name, _) in enumerate(experiments, 1):
        print(f"{idx}. {display_name}")

    print(f"\n{len(experiments) + 1}. EXIT")
    print("=" * 64)

    while True:
        try:
            choice = input("\nSelect an Xperiment (enter number): ").strip()
            choice_num = int(choice)

            if choice_num == len(experiments) + 1:
                print("Exiting...")
                sys.exit(0)

            if 1 <= choice_num <= len(experiments):
                return experiments[choice_num - 1]
            else:
                print(
                    f"Invalid choice. Please enter a number between 1 and {len(experiments) + 1}"
                )
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def show_menu_interactive(experiments):
    """
    Display an interactive menu using arrow keys for Xperiment selection.
    Only available on Unix-like systems (Linux, macOS).

    Args:
        experiments: List of tuples containing (display_name, file_path)

    Returns:
        tuple: (display_name, file_path) of the selected xperiment
    """
    if not HAS_INTERACTIVE_MENU:
        # Fallback to simple menu if interactive menu not available
        return show_menu_simple(experiments)

    # Get version from source (works with editable installs)
    try:
        from openwave import __version__

        pkg_version = __version__
    except ImportError:
        # Fallback to metadata if __version__ not available
        from importlib.metadata import version

        pkg_version = version("OPENWAVE")

    # Build menu options - all entries are selectable experiments
    menu_options = [display_name for display_name, _ in experiments]
    menu_options.append(None)  # Blank line separator
    menu_options.append("─── EXIT ───")
    exit_idx = len(experiments) + 1

    # Build title with proper formatting
    title_lines = [
        "",
        "=" * 64,
        f"OPENWAVE (v{pkg_version}) - Available XPERIMENTS",
        "(↑/↓ navigate, ENTER selects)",
        "=" * 64,
    ]

    terminal_menu = TerminalMenu(
        menu_options,
        title="\n".join(title_lines),
        menu_cursor="  ",
        menu_cursor_style=("fg_green", "bold"),
        menu_highlight_style=("bg_green", "fg_black"),
        cycle_cursor=True,
        skip_empty_entries=True,
    )

    choice_idx = terminal_menu.show()

    if choice_idx is None or choice_idx == exit_idx:
        print("Exiting...")
        sys.exit(0)

    return experiments[choice_idx]


def run_experiment(display_name, file_path):
    """
    Run the selected Xperiment file or open a URL.

    Args:
        display_name: Display name of the xperiment
        file_path: Path to the xperiment Python file, or a URL

    Returns:
        int: The return code from the experiment process
    """
    # Handle welcome URL specially
    if file_path == WELCOME_URL:
        print(f"\n{'=' * 64}")
        print("Opening welcome in your default browser...")
        print(f"{'=' * 64}\n")
        webbrowser.open(WELCOME_URL)
        print("Done!")
        return 0

    print(f"\n{'=' * 64}")
    print(f"Running XPERIMENT:")
    print(f"{display_name}")
    print(f"{'=' * 64}\n")

    try:
        # Run the xperiment using subprocess
        result = subprocess.run(
            [sys.executable, file_path],
            env=os.environ.copy(),
        )
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        return 0
    except Exception as e:
        print(f"\nError running XPERIMENT: {e}")
        return 1


def main():
    """
    Main entry point for the OpenWave CLI.

    This function is called when running 'openwave -x' from the command line.
    Runs the selected xperiment and exits when it closes.
    """
    # Get list of available experiments
    experiments = get_experiments_list()

    if not experiments:
        print("No Xperiments found in the xperiments directory.")
        sys.exit(1)

    # Show interactive menu and get user selection
    display_name, file_path = show_menu_interactive(experiments)

    # Run the selected xperiment
    returncode = run_experiment(display_name, file_path)

    # Exit after xperiment closes
    print(f"\n{'=' * 64}")
    print(f"XPERIMENT closed (exit code: {returncode})")
    print(f"{'=' * 64}\n")

    sys.exit(returncode)


def cli_main():
    """
    Main entry point for the 'openwave' command.

    Handles command-line arguments and routes to appropriate functionality.
    """
    parser = argparse.ArgumentParser(
        description="OpenWave - Subatomic Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-x",
        "--xperiments",
        action="store_true",
        help="launch the xperiments selector",
    )

    args = parser.parse_args()

    if args.xperiments:
        main()
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
