#!/usr/bin/env python3
"""
AWS Bedrock Model Testing Demonstrator

Tests Amazon Nova and OpenAI GPT-5 models available through AWS Bedrock.
"""

import json
import sys
import time

import boto3
from botocore.exceptions import ClientError
from openai import APIError, AuthenticationError, OpenAI

MODEL_CATALOG = {
    "amazon.nova-micro-v1:0": {"name": "Amazon Nova Micro", "type": "Text Generation", "context": "128K tokens", "description": "Text-only model optimized for lowest latency and cost"},
    "amazon.nova-lite-v1:0": {"name": "Amazon Nova Lite", "type": "Multimodal Generation", "context": "300K tokens", "description": "Cost-efficient multimodal (text, image, video)"},
    "amazon.nova-pro-v1:0": {"name": "Amazon Nova Pro", "type": "Multimodal Generation", "context": "300K tokens", "description": "High-performance multimodal for complex reasoning"},
    "amazon.nova-premier-v1:0": {"name": "Amazon Nova Premier", "type": "Multimodal Generation", "context": "1M tokens", "description": "Most capable model for complex multi-step reasoning"},
    "amazon.nova-2-lite-v1:0": {
        "name": "Amazon Nova 2 Lite",
        "type": "Multimodal Generation",
        "context": "1M tokens",
        "description": "Advanced reasoning with extended thinking, web grounding, code interpreter",
    },
    "amazon.nova-2-multimodal-embeddings-v1:0": {
        "name": "Nova 2 Multimodal Embeddings",
        "type": "Embeddings",
        "context": "8,172 tokens",
        "description": "Unified embedding model for text, documents, images, video, audio",
    },
    "openai.gpt-5.4": {"name": "OpenAI GPT-5.4", "type": "Text Generation", "context": "~1M tokens", "description": "Affordable, capable model for standard coding and professional tasks"},
    "openai.gpt-5.5": {"name": "OpenAI GPT-5.5", "type": "Text Generation", "context": "~1M tokens", "description": "Advanced reasoning for coding and professional tasks"},
    "openai.gpt-5.6": {"name": "OpenAI GPT-5.6", "type": "Text Generation", "context": "1.05M tokens", "description": "Latest flagship with Sol/Terra/Luna variants"},
}


AWS_REGION = "us-east-1"
TEST_PROMPT = "Explain AWS Bedrock in one sentence."
EMBEDDING_TEST_INPUT = "AWS Bedrock foundation model service"


class ResultsCollector:
    def __init__(self):
        self.successful = []
        self.failed = []
        self.total_time = 0.0

    def add_success(self, model_id: str, time_taken: float):
        self.successful.append(model_id)
        self.total_time += time_taken

    def add_failure(self, model_id: str, error: str):
        self.failed.append({"model": model_id, "error": error})


def print_banner():
    """Print welcome banner with model catalog."""
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║         AWS Bedrock Model Testing Demonstrator                   ║")
    print(f"║         Region: {AWS_REGION:<48}║")
    print(f"║         Total Models: {len(MODEL_CATALOG):<43}║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    print("═" * 67)
    print("📋 MODEL CATALOG")
    print("═" * 67)

    for idx, (model_id, info) in enumerate(MODEL_CATALOG.items(), 1):
        print(f"\n[{idx}] {model_id}")
        print(f"    Name: {info['name']}")
        print(f"    Type: {info['type']}")
        print(f"    Context: {info['context']}")
        print(f"    Description: {info['description']}")

    print("\n" + "═" * 67 + "\n")


def handle_error(error: Exception, model_id: str) -> str:
    """Handle and format error messages."""
    if isinstance(error, ClientError):
        error_code = error.response["Error"]["Code"]
        error_message = error.response["Error"]["Message"]

        if error_code == "AccessDeniedException":
            return "AccessDeniedException - Model access not enabled\n💡 Fix: Enable model access in AWS Bedrock console"
        elif error_code in ["ThrottlingException", "ServiceQuotaExceededException"]:
            return f"Rate limit exceeded - {error_message}\n💡 Fix: Consider implementing retry logic or request quota increase"
        elif error_code == "ValidationException":
            return f"ValidationException - {error_message}\n💡 Fix: Check model parameters or model ID format"
        elif error_code == "ResourceNotFoundException":
            return f"ResourceNotFoundException - Model not available in {AWS_REGION}\n💡 Fix: Check model availability in this region"
        else:
            return f"{error_code} - {error_message}"

    elif isinstance(error, AuthenticationError):
        return f"Authentication failed - {str(error)}\n💡 Fix: Verify AWS credentials and Bedrock access"

    elif isinstance(error, APIError):
        return f"API Error - {str(error)}"

    else:
        return f"Unexpected error - {str(error)}"


def run_nova_generation_models(results: ResultsCollector):
    """Test Amazon Nova generation models using Converse API."""
    print("═" * 67)
    print("🧪 TESTING NOVA GENERATION MODELS")
    print("═" * 67 + "\n")

    generation_models = ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0", "amazon.nova-premier-v1:0", "amazon.nova-2-lite-v1:0"]

    try:
        bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
    except Exception as e:
        print(f"✗ Failed to initialize Bedrock client: {e}\n")
        for model_id in generation_models:
            results.add_failure(model_id, f"Client initialization failed: {e}")
        return

    for idx, model_id in enumerate(generation_models, 1):
        print(f"[{idx}/{len(generation_models)}] Testing {model_id}...")

        try:
            start_time = time.time()

            response = bedrock.converse(modelId=model_id, messages=[{"role": "user", "content": [{"text": TEST_PROMPT}]}], inferenceConfig={"temperature": 0.7, "maxTokens": 512})

            elapsed_time = time.time() - start_time

            response_text = response["output"]["message"]["content"][0]["text"]
            truncated_text = response_text[:150] + "..." if len(response_text) > 150 else response_text

            input_tokens = response["usage"]["inputTokens"]
            output_tokens = response["usage"]["outputTokens"]

            print(f"✓ Response: {truncated_text}")
            print(f"⏱️  Latency: {elapsed_time:.2f}s")
            print(f"📊 Input tokens: {input_tokens}, Output tokens: {output_tokens}\n")

            results.add_success(model_id, elapsed_time)

        except Exception as e:
            error_msg = handle_error(e, model_id)
            print(f"✗ Error: {error_msg}\n")
            results.add_failure(model_id, error_msg)


def run_nova_embeddings(results: ResultsCollector):
    """Test Amazon Nova Multimodal Embeddings model."""
    print("═" * 67)
    print("🔢 TESTING EMBEDDINGS MODEL")
    print("═" * 67 + "\n")

    model_id = "amazon.nova-2-multimodal-embeddings-v1:0"

    try:
        bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
    except Exception as e:
        print(f"✗ Failed to initialize Bedrock client: {e}\n")
        results.add_failure(model_id, f"Client initialization failed: {e}")
        return

    print(f"[1/1] Testing {model_id}...")

    try:
        start_time = time.time()

        request_body = {"inputText": EMBEDDING_TEST_INPUT, "embeddingConfig": {"outputEmbeddingLength": 1024}}

        response = bedrock.invoke_model(modelId=model_id, body=json.dumps(request_body), contentType="application/json", accept="application/json")

        elapsed_time = time.time() - start_time

        response_body = json.loads(response["body"].read())

        if "embedding" in response_body:
            embedding_dims = len(response_body["embedding"])
            print("✓ Embedding generated successfully")
            print(f"📐 Dimensions: {embedding_dims}")
            print(f"⏱️  Latency: {elapsed_time:.2f}s\n")

            results.add_success(model_id, elapsed_time)
        else:
            error_msg = "Unexpected response format - no embedding found"
            print(f"✗ Error: {error_msg}\n")
            results.add_failure(model_id, error_msg)

    except Exception as e:
        error_msg = handle_error(e, model_id)
        print(f"✗ Error: {error_msg}\n")
        results.add_failure(model_id, error_msg)


def run_openai_models(results: ResultsCollector):
    """Test OpenAI GPT-5 models via Bedrock."""
    print("═" * 67)
    print("🤖 TESTING OPENAI GPT-5 MODELS")
    print("═" * 67 + "\n")

    openai_models = ["openai.gpt-5.4", "openai.gpt-5.5", "openai.gpt-5.6"]

    base_url = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/openai/v1"

    try:
        client = OpenAI(api_key="PLACEHOLDER", base_url=base_url)
    except Exception as e:
        print(f"✗ Failed to initialize OpenAI client: {e}\n")
        for model_id in openai_models:
            results.add_failure(model_id, f"Client initialization failed: {e}")
        return

    for idx, model_id in enumerate(openai_models, 1):
        print(f"[{idx}/{len(openai_models)}] Testing {model_id}...")

        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model=model_id, messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": TEST_PROMPT}], temperature=0.7, max_tokens=512
            )

            elapsed_time = time.time() - start_time

            response_text = response.choices[0].message.content
            truncated_text = response_text[:150] + "..." if len(response_text) > 150 else response_text

            print(f"✓ Response: {truncated_text}")
            print(f"⏱️  Latency: {elapsed_time:.2f}s\n")

            results.add_success(model_id, elapsed_time)

        except Exception as e:
            error_msg = handle_error(e, model_id)
            print(f"✗ Error: {error_msg}\n")
            results.add_failure(model_id, error_msg)


def print_summary(results: ResultsCollector):
    """Print test summary."""
    print("═" * 67)
    print("📊 SUMMARY")
    print("═" * 67 + "\n")

    total_tests = len(results.successful) + len(results.failed)
    success_count = len(results.successful)
    failure_count = len(results.failed)

    print(f"✓ Successful: {success_count}/{total_tests} models")
    print(f"✗ Failed: {failure_count}/{total_tests} models")

    if results.failed:
        print("\nFailed Models:")
        for failure in results.failed:
            print(f"  • {failure['model']}")
            error_lines = failure["error"].split("\n")
            for line in error_lines:
                if line.strip():
                    print(f"    {line}")

    print(f"\nTotal Execution Time: {results.total_time:.2f}s")
    print("\n" + "═" * 67 + "\n")


def main():
    """Main execution function."""
    print_banner()

    results = ResultsCollector()

    run_nova_generation_models(results)
    run_nova_embeddings(results)
    run_openai_models(results)

    print_summary(results)

    if results.failed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
