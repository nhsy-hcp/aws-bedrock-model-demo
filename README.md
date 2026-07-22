# AWS Bedrock Model Testing Demonstrator

Python scripts to test AWS Bedrock models and AgentCore:
- Amazon Nova v1 family (Micro, Lite, Pro)
- Amazon Nova 2 models (Lite, Multimodal Embeddings)
- AgentCore Harness demo using Amazon Nova Pro

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
# Run the Nova model demo
task demo

# Run the AgentCore Harness demo
task demo:agentcore

# Run tests
task test

# Run linting and formatting
task lint
```

## AgentCore Harness Demo

Creates a managed Harness backed by Amazon Nova Pro, invokes it with a prompt, streams the response, then keeps the harness for reuse on subsequent runs.

### One-time setup

Run this once to create the required IAM execution role:

```bash
task demo:agentcore:setup
```

This creates `bedrock-agentcore-demo-role` in your AWS account. The role ARN is then auto-derived from your active AWS credentials on every subsequent run — no environment variables needed.

### Running the demo

```bash
task demo:agentcore
```

The harness is reused on subsequent runs (no re-deploy). To clean up:

```bash
task demo:agentcore:delete
```

## Development

### Running Tests

```bash
task test
```

### Linting and Formatting

```bash
task lint
```

## Troubleshooting

### AccessDeniedException
Enable model access in AWS Bedrock console → Model access

### AgentCore demo fails with AccessDeniedException
Run `task demo:agentcore:setup` to create or update the IAM execution role.

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
