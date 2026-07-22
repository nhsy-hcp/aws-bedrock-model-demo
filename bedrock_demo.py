#!/usr/bin/env python3
"""
AWS Bedrock Model Testing Demonstrator

Tests Amazon Nova models available through AWS Bedrock.
"""

import json
import sys
import time

import boto3
from botocore.exceptions import ClientError

MODEL_CATALOG = {
    "amazon.nova-micro-v1:0": {
        "name": "Amazon Nova Micro",
        "type": "Text Generation",
        "context": "128K tokens | Max output: 5K",
        "description": "Amazon's fastest text-only model optimized for speed and low cost. Excels at summarization, translation, and classification. Knowledge cutoff: Oct 2024. Launched: Dec 2024.",
    },
    "amazon.nova-lite-v1:0": {
        "name": "Amazon Nova Lite",
        "type": "Multimodal Generation",
        "context": "300K tokens | Max output: 5K",
        "description": "Amazon's low-cost multimodal model processing text, images, and video. Ideal for document analysis, visual Q&A, and cost-sensitive workloads. Knowledge cutoff: Oct 2024. Launched: Dec 2024.",
    },
    "amazon.nova-pro-v1:0": {
        "name": "Amazon Nova Pro",
        "type": "Multimodal Generation",
        "context": "300K tokens | Max output: 5K",
        "description": "Amazon's balanced multimodal model offering strong accuracy, speed, and cost across text, images, and video. Best price-performance for most enterprise tasks. Knowledge cutoff: Oct 2024. Launched: Dec 2024.",
    },
    "us.amazon.nova-2-lite-v1:0": {
        "name": "Amazon Nova 2 Lite",
        "type": "Multimodal Generation",
        "context": "1M tokens | Max output: 64K",
        "description": "Cost-efficient multimodal model for automation, document processing, and customer support. Features extended thinking, web grounding, code interpreter, and optimized agent workflows. Knowledge cutoff: Oct 2025. Launched: Dec 2025.",
    },
    "amazon.nova-2-multimodal-embeddings-v1:0": {
        "name": "Amazon Nova Multimodal Embeddings",
        "type": "Embeddings",
        "context": "Dimensions: 3072 / 1024 / 384 / 256",
        "description": "Unified embedding model converting text, images, documents, video, and audio into a single vector space. Enables cross-modal retrieval, multimodal semantic search, and agentic RAG. Launched: Oct 2025.",
    },
}

GENERATION_MODELS = [k for k, v in MODEL_CATALOG.items() if v["type"] != "Embeddings"]
EMBEDDING_MODEL = next(k for k, v in MODEL_CATALOG.items() if v["type"] == "Embeddings")

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


def get_bedrock_client():
    """Create and return a Bedrock runtime client."""
    return boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)


def print_banner():
    """Print welcome banner with model catalog."""
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║         AWS Bedrock Nova & AgentCore Demo                        ║")
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
            return f"ResourceNotFoundException - {error_message}\n💡 Fix: Enable model access in the AWS Bedrock console (cross-region inference profiles require explicit access)"
        else:
            return f"{error_code} - {error_message}"
    else:
        return f"Unexpected error - {str(error)}"


def run_nova_generation_models(results: ResultsCollector, bedrock):
    """Test Amazon Nova generation models using Converse API."""
    print("═" * 67)
    print("🧪 TESTING NOVA GENERATION MODELS")
    print("═" * 67 + "\n")

    for idx, model_id in enumerate(GENERATION_MODELS, 1):
        print(f"[{idx}/{len(GENERATION_MODELS)}] Testing {model_id}...")

        try:
            start_time = time.time()

            response = bedrock.converse(modelId=model_id, messages=[{"role": "user", "content": [{"text": TEST_PROMPT}]}], inferenceConfig={"temperature": 0.7, "maxTokens": 512})

            elapsed_time = time.time() - start_time

            response_text = response["output"]["message"]["content"][0]["text"]
            input_tokens = response["usage"]["inputTokens"]
            output_tokens = response["usage"]["outputTokens"]

            print(f"✓ Response: {response_text}")
            print(f"⏱️  Latency: {elapsed_time:.2f}s")
            print(f"📊 Input tokens: {input_tokens}, Output tokens: {output_tokens}\n")

            results.add_success(model_id, elapsed_time)

        except Exception as e:
            error_msg = handle_error(e, model_id)
            print(f"✗ Error: {error_msg}\n")
            results.add_failure(model_id, error_msg)


def run_nova_embeddings(results: ResultsCollector, bedrock):
    """Test Amazon Nova Multimodal Embeddings model."""
    print("═" * 67)
    print("🔢 TESTING EMBEDDINGS MODEL")
    print("═" * 67 + "\n")

    model_id = EMBEDDING_MODEL

    print(f"[1/1] Testing {model_id}...")

    try:
        start_time = time.time()

        request_body = {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {"embeddingPurpose": "GENERIC_INDEX", "embeddingDimension": 1024, "text": {"truncationMode": "END", "value": EMBEDDING_TEST_INPUT}},
        }

        response = bedrock.invoke_model(modelId=model_id, body=json.dumps(request_body), contentType="application/json", accept="application/json")

        elapsed_time = time.time() - start_time

        response_body = json.loads(response["body"].read())

        embedding_key = "embedding" if "embedding" in response_body else "embeddings" if "embeddings" in response_body else None

        if embedding_key:
            embedding_dims = len(response_body[embedding_key])
            print("✓ Embedding generated successfully")
            print(f"📐 Dimensions: {embedding_dims}")
            print(f"⏱️  Latency: {elapsed_time:.2f}s\n")
            results.add_success(model_id, elapsed_time)
        else:
            error_msg = f"Unexpected response format - no embedding found. Response keys: {list(response_body.keys())}"
            print(f"✗ Error: {error_msg}\n")
            results.add_failure(model_id, error_msg)

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

    try:
        bedrock = get_bedrock_client()
    except Exception as e:
        print(f"✗ Failed to initialize Bedrock client: {e}\n")
        for model_id in MODEL_CATALOG:
            results.add_failure(model_id, f"Client initialization failed: {e}")
        print_summary(results)
        sys.exit(1)

    run_nova_generation_models(results, bedrock)
    run_nova_embeddings(results, bedrock)

    print_summary(results)

    if results.failed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
