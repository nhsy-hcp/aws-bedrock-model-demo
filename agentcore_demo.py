#!/usr/bin/env python3
"""
AWS Bedrock AgentCore Harness Demo

Creates an AgentCore Harness backed by Amazon Nova Pro, invokes it with a prompt,
streams the response, then tears down the harness.

Prerequisites:
  export AGENTCORE_EXECUTION_ROLE_ARN=arn:aws:iam::<account>:role/<role-name>

Minimum IAM policy for the execution role trust policy:
  Principal: {"Service": "bedrock-agentcore.amazonaws.com"}

Minimum IAM policy for the execution role permissions:
  bedrock:InvokeModel on amazon.nova-pro-v1:0
  logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents
"""

import os
import sys
import time
import uuid

import boto3
from botocore.exceptions import ClientError

AWS_REGION = "us-east-1"
HARNESS_MODEL_ID = "amazon.nova-pro-v1:0"
HARNESS_NAME = "BedrockModelDemoHarness"
POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 300
TEST_PROMPT = "Explain AWS Bedrock AgentCore in two sentences, focusing on what problems it solves for developers."


EXECUTION_ROLE_NAME = "bedrock-agentcore-demo-role"


def get_execution_role_arn() -> str:
    """Resolve execution role ARN from env var or derive from current account."""
    role_arn = os.environ.get("AGENTCORE_EXECUTION_ROLE_ARN", "")
    if role_arn:
        return role_arn

    try:
        sts = boto3.client("sts", region_name=AWS_REGION)
        account_id = sts.get_caller_identity()["Account"]
        derived = f"arn:aws:iam::{account_id}:role/{EXECUTION_ROLE_NAME}"
        print(f"    Role: {derived} (auto-derived)")
        return derived
    except Exception as e:
        print(f"\n✗ Could not determine AWS account ID: {e}")
        print("  Run: task demo:agentcore:setup")
        print(f"  Then: export AGENTCORE_EXECUTION_ROLE_ARN=arn:aws:iam::<account>:role/{EXECUTION_ROLE_NAME}")
        sys.exit(1)


def print_banner():
    """Print welcome banner."""
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║         AWS Bedrock AgentCore Harness Demo                       ║")
    print(f"║         Region: {AWS_REGION:<48}║")
    print(f"║         Model:  {HARNESS_MODEL_ID:<48}║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")


def handle_error(error: Exception) -> str:
    """Handle and format error messages."""
    if isinstance(error, ClientError):
        code = error.response["Error"]["Code"]
        message = error.response["Error"]["Message"]
        if code == "AccessDeniedException":
            return f"AccessDeniedException - {message}\n💡 Fix: Check IAM permissions on the execution role and your caller credentials"
        elif code == "ValidationException":
            return f"ValidationException - {message}\n💡 Fix: Check the harness configuration parameters"
        elif code == "ResourceNotFoundException":
            return f"ResourceNotFoundException - {message}\n💡 Fix: Check AgentCore is available in {AWS_REGION}"
        elif code in ["ThrottlingException", "ServiceQuotaExceededException"]:
            return f"Rate limit exceeded - {message}\n💡 Fix: Wait and retry, or request a quota increase"
        else:
            return f"{code} - {message}"
    return f"Unexpected error - {str(error)}"


def get_or_create_harness(control_client, role_arn: str) -> tuple[str, str, bool]:
    """Return (harness_arn, harness_id, created). Reuses existing harness if found and alive."""
    response = control_client.list_harnesses()
    for h in response.get("harnesses", []):
        if h["harnessName"] == HARNESS_NAME:
            try:
                detail = control_client.get_harness(harnessId=h["harnessId"])
                status = detail["harness"]["status"]
                if status not in ("DELETING", "DELETE_FAILED"):
                    print(f"[1/4] Reusing existing harness: {HARNESS_NAME} (status: {status})")
                    print(f"    ARN: {h['arn']}")
                    return h["arn"], h["harnessId"], False
            except ClientError:
                pass

    print(f"[1/4] Creating harness: {HARNESS_NAME}...")
    response = control_client.create_harness(
        harnessName=HARNESS_NAME,
        executionRoleArn=role_arn,
        model={
            "bedrockModelConfig": {
                "modelId": HARNESS_MODEL_ID,
                "maxTokens": 512,
                "temperature": 0.7,
            }
        },
        systemPrompt=[{"text": "You are a helpful AWS assistant. Be concise and accurate."}],
    )
    harness_arn = response["harness"]["arn"]
    harness_id = response["harness"]["harnessId"]
    print(f"    ARN: {harness_arn}")
    return harness_arn, harness_id, True


def wait_for_ready(control_client, harness_id: str) -> None:
    """Poll until harness status is READY."""
    print("[2/4] Waiting for harness to be READY", end="", flush=True)
    deadline = time.time() + POLL_TIMEOUT_SECONDS

    while time.time() < deadline:
        response = control_client.get_harness(harnessId=harness_id)
        status = response["harness"]["status"]

        if status == "READY":
            print(" ✓")
            return
        elif status in ("FAILED", "DELETE_FAILED"):
            print(f" ✗\n✗ Harness entered status: {status}")
            sys.exit(1)

        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL_SECONDS)

    print(" ✗")
    raise TimeoutError(f"Harness did not become READY within {POLL_TIMEOUT_SECONDS}s")


def invoke_harness(runtime_client, harness_arn: str) -> tuple[str, float]:
    """Invoke the harness and stream the response. Returns (response_text, latency)."""
    print("[3/4] Invoking harness...")
    print(f"    Prompt: {TEST_PROMPT}\n")
    print("    Response:")
    print("    " + "─" * 60)

    session_id = str(uuid.uuid4())
    start_time = time.time()

    response = runtime_client.invoke_harness(
        harnessArn=harness_arn,
        runtimeSessionId=session_id,
        messages=[{"role": "user", "content": [{"text": TEST_PROMPT}]}],
    )

    full_text = []
    print("    ", end="", flush=True)

    input_tokens = 0
    output_tokens = 0

    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                chunk = delta["text"]
                full_text.append(chunk)
                print(chunk, end="", flush=True)
        elif "metadata" in event:
            usage = event["metadata"].get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
        elif "runtimeClientError" in event:
            error_msg = event["runtimeClientError"].get("message", "Unknown stream error")
            print(f"\n✗ Stream error: {error_msg}")
            sys.exit(1)
        elif "internalServerException" in event:
            error_msg = event["internalServerException"].get("message", "Internal server error")
            print(f"\n✗ Stream error: {error_msg}")
            sys.exit(1)
        elif "validationException" in event:
            error_msg = event["validationException"].get("message", "Validation error")
            print(f"\n✗ Stream error: {error_msg}")
            sys.exit(1)

    elapsed = time.time() - start_time
    print("\n    " + "─" * 60)
    print(f"\n⏱️  Latency: {elapsed:.2f}s")
    print(f"📊 Input tokens: {input_tokens}, Output tokens: {output_tokens}")
    return "".join(full_text), elapsed


def delete_harness(control_client, harness_id: str) -> None:
    """Delete the harness."""
    print(f"[4/4] Deleting harness {harness_id}...")
    try:
        control_client.delete_harness(harnessId=harness_id)
        print("    ✓ Harness deleted")
    except Exception as e:
        print(f"    ⚠ Could not delete harness: {handle_error(e)}")
        print(f"    Manual cleanup: aws bedrock-agentcore-control delete-harness --harness-id {harness_id}")


def main():
    """Main execution function."""
    delete = "--delete" in sys.argv

    print_banner()

    role_arn = get_execution_role_arn()

    control_client = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)
    runtime_client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)

    harness_arn = None
    harness_id = None
    created = False

    try:
        harness_arn, harness_id, created = get_or_create_harness(control_client, role_arn)
        wait_for_ready(control_client, harness_id)
        response_text, latency = invoke_harness(runtime_client, harness_arn)

        print("\n" + "═" * 67)
        print("📊 SUMMARY")
        print("═" * 67)
        print(f"✓ Harness:      {HARNESS_NAME} ({'new' if created else 'reused'})")
        print(f"✓ Harness ARN:  {harness_arn}")
        print(f"✓ Model:        {HARNESS_MODEL_ID}")
        print(f"✓ Latency:      {latency:.2f}s")
        print("═" * 67 + "\n")

        if delete:
            delete_harness(control_client, harness_id)
        else:
            print("💡 Harness kept for reuse. To delete: task demo:agentcore:delete")

    except TimeoutError as e:
        print(f"\n✗ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {handle_error(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
