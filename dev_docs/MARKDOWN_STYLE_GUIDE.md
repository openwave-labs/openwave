# Markdown Style Guide

## Overview

This guide ensures consistent markdown formatting across all OpenWave documentation.

All `.md` files must comply with these standards to pass linting checks.

## Required Linting Rules

### MD022: Blank Lines Around Headings

Always add a blank line after headings before content starts.

**Good:**

```markdown
## Heading

Content starts here.
```

**Bad:**

```markdown
## Heading
Content starts here.
```

### MD047: Single Trailing Newline

Every markdown file must end with exactly one newline character.

**Good:**

```markdown
Last line of content.

```

**Bad:**

```markdown
Last line of content.```

### MD032: Blank Lines Around Lists

Lists should be surrounded by blank lines.

**Good:**

```markdown
Some text before the list.

- Item 1
- Item 2

Some text after the list.
```

**Bad:**

```markdown
Some text before the list.
- Item 1
- Item 2
Some text after the list.
```

## Additional Best Practices

### Code Blocks

Always specify the language for syntax highlighting:

```python
# Python code
def example():
    pass
```

### Line Length

While not enforced by linter, consider keeping lines under 120 characters for better readability in code editors.

## Linting Tools

The project uses `markdownlint` for enforcing these standards. Common issues:

- `MD022/blanks-around-headings`: Add blank lines after headings
- `MD047/single-trailing-newline`: Ensure file ends with single newline
- `MD032/blanks-around-lists`: Add blank lines around lists

## Quick Checklist

Before committing any `.md` file:

- [ ] All headings have blank lines after them
- [ ] Lists have blank lines before and after
- [ ] File ends with exactly one newline
- [ ] Code blocks specify language
- [ ] Links use descriptive text

## VS Code Configuration

If using VS Code with markdownlint extension, these rules are automatically highlighted. The project's `.markdownlint.json` (if present) contains the specific configuration.

## References

- [Markdownlint Rules](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)
- [CommonMark Specification](https://commonmark.org/)
