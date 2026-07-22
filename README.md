# AWS Bedrock Model Testing Demonstrator

Simple Python script to test AWS Bedrock models including:
- Amazon Nova v1 family (Micro, Lite, Pro)
- Amazon Nova 2 models (Lite, Multimodal Embeddings)

## Prerequisites

1. AWS credentials configured with Bedrock access
2. Model access enabled in AWS Bedrock console (us-east-1)
3. Python 3.11+ and `uv` installed

## Installation

```bash
cd bedrock-model-demo
uv sync
```

## Usage

```bash
AWS_PROFILE=your-profile-name uv run python bedrock_demo.py
```

Or use the Taskfile:

```bash
# Run the demo
task demo

# Run tests
task test

# Run linting
task lint

# Format code
task fmt
```

## Development

### Running Tests

```bash
task test
```

### Linting

```bash
task lint
```

### Code Formatting

```bash
task fmt
```

## Troubleshooting

### AccessDeniedException
Enable model access in AWS Bedrock console → Model access

### Nova Premier not available
Nova Premier has been removed from this demo (Legacy model, EOL Sep 14, 2026).

### Authentication errors
Verify AWS profile is set correctly:
```bash
echo $AWS_PROFILE
aws sts get-caller-identity
```

## Output

The script displays:
- Model catalog with descriptions and context windows
- Test results for each model
- Response text and latency metrics
- Summary with success/failure counts

## Testing

The project includes comprehensive unit tests using pytest with mocked AWS responses.

Run tests with:
```bash
task test
```

Or directly with pytest:
```bash
uv run pytest -v
```
