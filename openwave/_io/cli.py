"""
OpenWave CLI - Command Line Interface for running Xperiments.

This module provides the command-line entry point for OpenWave,
allowing users to interactively select and run Xperiments.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

from simple_term_menu import TerminalMenu


def get_experiments_list():
    """
    Get a list of available Xperiment files from the xperiments directory.

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

    # Get all Python files in the xperiments directory
    experiments = []
    for file_path in sorted(xperiments_dir.glob("*.py")):
        if file_path.name.startswith("__"):
            continue  # Skip __init__.py and similar files

        # Read the first few lines to get the description
        description = ""
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
                # Look for description in docstring
                if len(lines) > 1 and '"""' in lines[0]:
                    for line in lines[1:6]:
                        if '"""' in line:
                            break
                        if line.strip() and not line.strip().startswith("XPERIMENT:"):
                            description = line.strip()
                            break
                        elif line.strip().startswith("XPERIMENT:"):
                            description = line.strip().replace("XPERIMENT:", "").strip()
                            break
        except Exception:
            pass

        # Create display name
        name = file_path.stem.replace("_", " ").title()
        if description:
            display_name = f"{name} - {description}"
        else:
            display_name = name

        experiments.append((display_name, str(file_path)))

    return experiments


def show_menu_simple(experiments):
    """
    Display a simple numbered menu for Xperiment selection.

    Args:
        experiments: List of tuples containing (display_name, file_path)

    Returns:
        str: Path to the selected xperiment file
    """
    print("\n" + "=" * 64)
    print("OpenWave - Available Xperiments")
    print("=" * 64)

    for idx, (display_name, _) in enumerate(experiments, 1):
        print(f"{idx}. {display_name}")

    print(f"{len(experiments) + 1}. Exit")
    print("=" * 70)

    while True:
        try:
            choice = input("\nSelect an Xperiment (enter number): ").strip()
            choice_num = int(choice)

            if choice_num == len(experiments) + 1:
                print("Exiting...")
                sys.exit(0)

            if 1 <= choice_num <= len(experiments):
                return experiments[choice_num - 1][1]
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

    Args:
        experiments: List of tuples containing (display_name, file_path)

    Returns:
        str: Path to the selected xperiment file
    """
    # Create menu options
    menu_options = [display_name for display_name, _ in experiments]
    menu_options.append("EXIT")

    terminal_menu = TerminalMenu(
        menu_options,
        title="\nOPENWAVE - Available Xperiments\n(↑/↓ to navigate, ENTER to select)",
        menu_cursor="→ ",
        menu_cursor_style=("fg_green", "bold"),
        menu_highlight_style=("bg_green", "fg_black"),
    )

    choice_idx = terminal_menu.show()

    if choice_idx is None or choice_idx == len(experiments):
        print("Exiting...")
        sys.exit(0)

    return experiments[choice_idx][1]


def run_experiment(file_path):
    """
    Run the selected Xperiment file.

    Args:
        file_path: Path to the xperiment Python file
    """
    print(f"\n{'=' * 64}")
    print(f"Running XPERIMENT: {Path(file_path).stem}")
    print(f"{'=' * 64}\n")

    try:
        # Run the xperiment using subprocess
        result = subprocess.run(
            [sys.executable, file_path],
            env=os.environ.copy(),
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError running XPERIMENT: {e}")
        sys.exit(1)


def main():
    """
    Main entry point for the OpenWave CLI.

    This function is called when running 'openwave -x' from the command line.
    """
    # Check if -x flag was provided (handled by entry point name)
    experiments = get_experiments_list()

    if not experiments:
        print("No Xperiments found in the xperiments directory.")
        sys.exit(1)

    # Try to use interactive menu, fall back to simple menu if not available
    selected_file = show_menu_interactive(experiments)

    # Run the selected xperiment
    run_experiment(selected_file)


def cli_main():
    """
    Main entry point for the 'openwave' command.

    Handles command-line arguments and routes to appropriate functionality.
    """
    parser = argparse.ArgumentParser(
        description="OpenWave - Quantum Physics Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-x",
        "--xperiments",
        action="store_true",
        help="Launch the xperiments selector",
    )

    args = parser.parse_args()

    if args.xperiments:
        main()
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
