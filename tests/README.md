# Test Suite for Gaze-Guided Scene Graph

This directory contains the test suite for the gaze-guided-scene-graph project.

## Structure

The test suite follows the same structure as the main package:

```
tests/
├── unit/                 # Unit tests for individual components
│   ├── config/           # Tests for configuration module
│   ├── datasets/         # Tests for dataset handling
│   │   └── egtea_gaze/   # Tests for EGTEA Gaze dataset
│   ├── graph/            # Tests for graph module
│   ├── models/           # Tests for model components
│   └── training/         # Tests for training components
├── integration/          # Integration tests for component interaction
├── resources/            # Shared test resources
│   ├── data/             # Test data files
│   └── fixtures.py       # Shared pytest fixtures
└── conftest.py           # Test configuration
```

## Running Tests

To run the complete test suite:

```bash
python -m pytest
```

To run only unit tests:

```bash
python -m pytest tests/unit
```

To run tests with real models (which are skipped by default):

```bash
python -m pytest --run-real-model
``` 