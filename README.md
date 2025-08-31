# Parallax

A lightweight Python library that enables efficient parallel execution of AI API calls.

## Installation

You can install Parallax using pip:

```bash
pip install -e .  # For development
# or
pip install parallax  # When published to PyPI
```

## Development

1. Clone the repository:
   ```bash
   git clone https://github.com/panuthep/Parallax.git
   cd Parallax
   ```

2. Set up a virtual environment and install development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
