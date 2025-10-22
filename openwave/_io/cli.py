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
from importlib.metadata import version, metadata


from simple_term_menu import TerminalMenu


def get_experiments_list():
    """
    Get a list of available Xperiment files from the xperiments directory.
    Recursively searches subdirectories, excluding _docs folders.

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

    # Dictionary to organize experiments by group
    experiments_by_group = {}

    # Recursively find all Python files, excluding _docs directories
    for file_path in xperiments_dir.rglob("*.py"):
        # Skip files in directories that start with underscore (like _docs)
        if any(part.startswith("_") for part in file_path.parts):
            continue

        # Skip __init__.py and similar files
        if file_path.name.startswith("__"):
            continue

        # Determine group (subdirectory name or "root" if directly in xperiments)
        relative_path = file_path.relative_to(xperiments_dir)
        if len(relative_path.parts) > 1:
            group = relative_path.parts[0]
        else:
            group = "root"

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

        # Add to group
        if group not in experiments_by_group:
            experiments_by_group[group] = []

        experiments_by_group[group].append((display_name, str(file_path), group))

    # Build flat list with tree structure display
    experiments = []
    sorted_group = sorted(experiments_by_group.keys())

    for group_idx, group in enumerate(sorted_group):
        # Sort experiments within each group
        group_experiments = sorted(experiments_by_group[group], key=lambda x: x[0])

        for idx, (display_name, file_path, _) in enumerate(group_experiments):
            # Format with tree structure
            if group == "root":
                formatted_name = display_name
            else:
                # Add group header for first item in group
                if idx == 0:
                    # Add blank line separator before group (except for first group)
                    if group_idx > 0:
                        experiments.append(("", None))  # Blank line separator

                    # Format group name as header
                    # Check if __init__.py exists in group folder
                    group_init_path = xperiments_dir / group / "__init__.py"
                    group_display = None

                    if group_init_path.exists():
                        try:
                            with open(group_init_path, "r") as f:
                                lines = f.readlines()
                                # Look for docstring (use only first line)
                                if len(lines) > 0 and '"""' in lines[0]:
                                    for line in lines[1:]:
                                        if '"""' in line:
                                            break
                                        stripped = line.strip()
                                        if stripped:
                                            group_display = stripped
                                            break
                        except Exception:
                            pass

                    # Fall back to formatted group name if no docstring found
                    if not group_display:
                        group_display = (
                            group.replace("__", ": ").replace("_", " ").replace("-", " ").title()
                        )

                    experiments.append((f"─── /{group_display}/ ───", None))  # group header

                # Indent all items under group
                formatted_name = f"  → {display_name}"

            experiments.append((formatted_name, file_path))

    return experiments


def show_menu_simple(experiments):
    """
    Display a simple numbered menu for Xperiment selection.

    Args:
        experiments: List of tuples containing (display_name, file_path)

    Returns:
        str: Path to the selected xperiment file
    """
    pkg_version = version("OPENWAVE")

    print("\n" + "=" * 64)
    print(f"OPENWAVE (v{pkg_version}) - Available XPERIMENTS")
    print("=" * 64)

    # Create numbered list of selectable experiments
    selectable_experiments = []
    display_idx = 1

    for display_name, file_path in experiments:
        if file_path is None:  # group header or separator
            print(f"\n{display_name}")
        else:
            print(f"{display_idx}. {display_name}")
            selectable_experiments.append((display_name, file_path))
            display_idx += 1

    print(f"\n{len(selectable_experiments) + 1}. Exit")
    print("=" * 64)

    while True:
        try:
            choice = input("\nSelect an Xperiment (enter number): ").strip()
            choice_num = int(choice)

            if choice_num == len(selectable_experiments) + 1:
                print("Exiting...")
                sys.exit(0)

            if 1 <= choice_num <= len(selectable_experiments):
                return selectable_experiments[choice_num - 1][1]
            else:
                print(
                    f"Invalid choice. Please enter a number between 1 and {len(selectable_experiments) + 1}"
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
    # Build menu with group headers marked as non-selectable
    menu_options = []
    file_path_map = {}  # Maps option index to file path
    option_idx = 0
    pkg_version = version("OPENWAVE")

    for display_name, file_path in experiments:
        menu_options.append(display_name if display_name else " ")  # Empty line or display name
        if file_path is not None:
            file_path_map[option_idx] = file_path
        option_idx += 1

    menu_options.append("EXIT")
    exit_idx = option_idx

    terminal_menu = TerminalMenu(
        menu_options,
        title=f"\n==========================================================================\nOPENWAVE (v{pkg_version}) - Available XPERIMENTS (↑/↓ navigate, ENTER selects)\n==========================================================================",
        menu_cursor="  ",
        menu_cursor_style=("fg_green", "bold"),
        menu_highlight_style=("bg_green", "fg_black"),
    )

    while True:
        choice_idx = terminal_menu.show()

        if choice_idx is None or choice_idx == exit_idx:
            print("Exiting...")
            sys.exit(0)

        # Check if this is a selectable experiment
        if choice_idx in file_path_map:
            return file_path_map[choice_idx]
        # If group header selected, continue loop to allow re-selection


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
        description="OpenWave - Subatomic Simulator",
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
