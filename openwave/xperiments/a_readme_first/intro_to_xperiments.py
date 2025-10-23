import os
import sys
import webbrowser
from pathlib import Path


def main():
    """Open intro.html in the default browser."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()

    # Build the path to intro.html
    html_file = script_dir / "intro.html"

    # Check if the file exists
    if not html_file.exists():
        print(f"Error: intro.html not found at {html_file}")
        sys.exit(1)

    # Convert to file:// URL for better cross-platform compatibility
    file_url = html_file.as_uri()

    print(f"Opening {html_file} in your default browser...")

    # Open in the default browser
    webbrowser.open(file_url)

    print("Done!")


if __name__ == "__main__":
    main()
