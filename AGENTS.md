# AGENTS.md

## Development Commands

This file provides guidance for AI agents working on the `bedrock-model-demo` project.

### Project Structure

```
bedrock-model-demo/
├── bedrock_demo.py          # Nova model demo script
├── agentcore_demo.py        # AgentCore Harness demo script
├── tests/
│   ├── test_bedrock_demo.py    # Unit tests for Nova demo
│   └── test_agentcore_demo.py  # Unit tests for AgentCore demo
├── pyproject.toml           # Python dependencies and configuration
├── Taskfile.yml             # Task automation
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── .gitignore               # Git ignore patterns
├── LICENSE                  # MPL-2.0 license
└── README.md                # User documentation
```

### Standard Development Workflow

#### 1. Environment Setup

```bash
# Initialize project (install dependencies)
task init
```

#### 2. Testing

```bash
# Run all tests
task test

# Run unit tests only
task test:unit

# Run tests with coverage report (minimum 80% required)
task test:coverage
```

#### 3. Code Quality

```bash
# Run linting checks and format code automatically (200 char line length)
task lint
```

#### 4. Pre-commit Hooks

```bash
# Update pre-commit hooks to latest versions
task hooks:update

# Run pre-commit hooks on all files
task hooks:run
```

#### 5. Running the Demos

```bash
# Run the Nova model demo (requires AWS credentials)
task demo

# One-time IAM role setup (creates bedrock-agentcore-demo-role)
task demo:agentcore:setup

# Run the AgentCore Harness demo - no env var needed after setup
task demo:agentcore

# Run AgentCore demo and delete harness after invocation
task demo:agentcore:run:delete

# Delete all demo harnesses
task demo:agentcore:delete

# Or run directly with AWS profile
AWS_PROFILE=your-profile-name task demo
```

#### 6. Cleanup

```bash
# Clean up temporary files and caches
task clean
```

### Code Standards

- **Line Length**: Maximum 200 characters
- **Code Style**: Enforced by Ruff
- **Test Coverage**: Minimum 80%
- **Pre-commit Hooks**: Must pass before committing
  - Trailing whitespace removal
  - End of file fixing
  - YAML/TOML validation
  - Large file checks
  - Merge conflict detection
  - Private key detection
  - Ruff linting and formatting
  - Shellcheck (scripts/*.sh)
  - Gitleaks (secret detection)

### Important Guidelines

#### Git Operations

- **NEVER** use `git add .` or `git add` with wildcards
- Always stage specific files explicitly
- **Always run the following before proposing a commit:**
  ```bash
  task lint
  task test
  task test:coverage
  ```
- Confirm with user before:
  - Staging files
  - Creating branches
  - Committing changes

#### Code Changes

- Run `task lint` before proposing changes
- Ensure `task test:coverage` passes with ≥80% coverage
- Follow existing code patterns and conventions
- Maintain 200 character line length limit

#### Testing

- Write tests for all new functionality
- Mock external AWS API calls
- Use pytest fixtures appropriately
- Test error handling paths

### AWS Configuration

The demo requires AWS credentials with Bedrock access:

```bash
# Set AWS profile
export AWS_PROFILE=your-profile-name

# Verify credentials
aws sts get-caller-identity

# Run demo
task demo
```

### Models Tested

The demo tests the following AWS Bedrock models:

**Amazon Nova v1 Models:**
- `amazon.nova-micro-v1:0` - Text-only, 128K context, lowest latency and cost (launched Dec 2024)
- `amazon.nova-lite-v1:0` - Multimodal (text/image/video), 300K context, cost-efficient (launched Dec 2024)
- `amazon.nova-pro-v1:0` - Multimodal, 300K context, best price-performance for enterprise tasks (launched Dec 2024)

**Amazon Nova 2 Models:**
- `us.amazon.nova-2-lite-v1:0` - Multimodal, 1M context, 64K output. Extended thinking, web grounding, code interpreter. Geo cross-region inference profile (launched Dec 2025)
- `amazon.nova-2-multimodal-embeddings-v1:0` - Unified embeddings for text, images, documents, video, audio. Dimensions: 3072/1024/384/256 (launched Oct 2025)

**AgentCore Harness:**
- Uses `amazon.nova-pro-v1:0` as the backing model
- Managed agent loop with streaming responses
- Role ARN auto-derived from active AWS credentials using `bedrock-agentcore-demo-role` — no env var required after `task demo:agentcore:setup`
- Harness is created once and reused across runs; deleted only via `task demo:agentcore:delete`
- Flow: get-or-create harness, poll until READY, invoke with streaming

### Troubleshooting

#### Test Failures

```bash
# Run tests with verbose output
task test:coverage

# Check specific test
uv run pytest -v tests/test_bedrock_demo.py::TestClass::test_method
```

#### Linting Issues

```bash
# Auto-fix linting and formatting issues
task lint

# Check specific files
uv run ruff check bedrock_demo.py --line-length=200
```

#### Coverage Below 80%

```bash
# View coverage report
task test:coverage
# Open htmlcov/index.html to see detailed coverage report
```

#### Pre-commit Hook Failures

```bash
# Run hooks individually
uv run pre-commit run trailing-whitespace --all-files
uv run pre-commit run ruff --all-files
uv run pre-commit run gitleaks --all-files

# Update hooks if outdated
task hooks:update
```

### Dependencies

Core dependencies are managed via `uv` and defined in `pyproject.toml`:

**Runtime:**
- boto3 ≥1.35.0 (AWS SDK)

**Development:**
- pytest ≥8.0.0 (Testing framework)
- pytest-cov ≥6.0.0 (Coverage reporting)
- pytest-mock ≥3.12.0 (Mocking utilities)
- ruff ≥0.6.0 (Linting and formatting)
- pre-commit ≥4.0.0 (Git hooks)

### License

This project is licensed under MPL-2.0. See the LICENSE file for details.

### Questions?

If you encounter issues or need clarification:
1. Check this AGENTS.md file
2. Review README.md for user documentation
3. Examine existing tests for patterns
4. Check Taskfile.yml for available commands
