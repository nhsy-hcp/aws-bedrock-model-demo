# AWS Bedrock Model Testing Demonstrator

Simple Python script to test AWS Bedrock models including:
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
AWS_PROFILE=your-profile-name uv run python bedrock_demo.py
```

Or use the Taskfile:

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

The AgentCore demo creates a managed Harness backed by Amazon Nova Pro, invokes it with a prompt, streams the response, then tears down the harness automatically.

### Prerequisites

Create an IAM role with the following trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
```

And attach a permissions policy including at minimum:
- `bedrock:InvokeModel`
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`

Then export the role ARN:

```bash
export AGENTCORE_EXECUTION_ROLE_ARN=arn:aws:iam::<account>:role/<role-name>
task demo:agentcore
```

### Taskfile commands

```bash
# One-time IAM role setup (creates bedrock-agentcore-demo-role)
task demo:agentcore:setup

# Export the printed role ARN, then run the demo
export AGENTCORE_EXECUTION_ROLE_ARN=arn:aws:iam::<account>:role/bedrock-agentcore-demo-role
task demo:agentcore

# Run demo and keep harness after invocation
task demo:agentcore:keep

# Delete all demo harnesses (cleanup after --keep)
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
