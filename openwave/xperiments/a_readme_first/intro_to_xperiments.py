import os
import sys
import webbrowser
from pathlib import Path


def main():
    """Open intro in the default browser."""

    url = "https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/a_readme_first/intro.md"
    print(f"Opening intro in your default browser...")

    # Open in the default browser
    webbrowser.open(url)

    print("Done!")


if __name__ == "__main__":
    main()
