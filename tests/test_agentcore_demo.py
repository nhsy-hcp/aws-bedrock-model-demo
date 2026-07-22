"""Unit tests for AWS Bedrock AgentCore Harness demo."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from agentcore_demo import AWS_REGION, HARNESS_MODEL_ID, TEST_PROMPT, create_harness, delete_harness, get_execution_role_arn, handle_error, invoke_harness, wait_for_ready


class TestGetExecutionRoleArn:
    """Test execution role ARN retrieval."""

    def test_returns_arn_when_set(self, monkeypatch):
        """Test that ARN is returned when env var is set."""
        monkeypatch.setenv("AGENTCORE_EXECUTION_ROLE_ARN", "arn:aws:iam::123456789012:role/MyRole")
        assert get_execution_role_arn() == "arn:aws:iam::123456789012:role/MyRole"

    def test_exits_when_not_set(self, monkeypatch, capsys):
        """Test that sys.exit(1) is called when env var is missing."""
        monkeypatch.delenv("AGENTCORE_EXECUTION_ROLE_ARN", raising=False)
        with pytest.raises(SystemExit) as exc:
            get_execution_role_arn()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "AGENTCORE_EXECUTION_ROLE_ARN" in captured.out


class TestHandleError:
    """Test error handling."""

    def test_access_denied(self):
        error = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "Denied"}}, "op")
        result = handle_error(error)
        assert "AccessDeniedException" in result
        assert "IAM permissions" in result

    def test_validation_exception(self):
        error = ClientError({"Error": {"Code": "ValidationException", "Message": "Bad param"}}, "op")
        result = handle_error(error)
        assert "ValidationException" in result

    def test_resource_not_found(self):
        error = ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}}, "op")
        result = handle_error(error)
        assert "ResourceNotFoundException" in result
        assert AWS_REGION in result

    def test_throttling(self):
        error = ClientError({"Error": {"Code": "ThrottlingException", "Message": "Too fast"}}, "op")
        result = handle_error(error)
        assert "Rate limit exceeded" in result

    def test_generic_client_error(self):
        error = ClientError({"Error": {"Code": "InternalServerError", "Message": "Boom"}}, "op")
        result = handle_error(error)
        assert "InternalServerError" in result

    def test_unexpected_error(self):
        error = Exception("Something broke")
        result = handle_error(error)
        assert "Unexpected error" in result
        assert "Something broke" in result


class TestCreateHarness:
    """Test harness creation."""

    def test_successful_create(self, capsys):
        """Test successful harness creation."""
        mock_client = MagicMock()
        mock_client.create_harness.return_value = {
            "harnessArn": "arn:aws:bedrock-agentcore:us-east-1:123456789012:harness/test-abc123",
            "harnessId": "test-abc123",
        }

        arn, harness_id = create_harness(mock_client, "arn:aws:iam::123456789012:role/MyRole")

        assert arn == "arn:aws:bedrock-agentcore:us-east-1:123456789012:harness/test-abc123"
        assert harness_id == "test-abc123"

        call_kwargs = mock_client.create_harness.call_args[1]
        assert call_kwargs["executionRoleArn"] == "arn:aws:iam::123456789012:role/MyRole"
        assert call_kwargs["model"]["bedrockModelConfig"]["modelId"] == HARNESS_MODEL_ID

        captured = capsys.readouterr()
        assert "Creating harness" in captured.out


class TestWaitForReady:
    """Test harness readiness polling."""

    def test_immediately_ready(self, capsys):
        """Test harness that is immediately READY."""
        mock_client = MagicMock()
        mock_client.get_harness.return_value = {"status": "READY"}

        wait_for_ready(mock_client, "test-abc123")

        mock_client.get_harness.assert_called_once_with(harnessId="test-abc123")

    def test_ready_after_polling(self, capsys):
        """Test harness that becomes READY after a few polls."""
        mock_client = MagicMock()
        mock_client.get_harness.side_effect = [
            {"status": "CREATING"},
            {"status": "CREATING"},
            {"status": "READY"},
        ]

        with patch("agentcore_demo.time.sleep"):
            wait_for_ready(mock_client, "test-abc123")

        assert mock_client.get_harness.call_count == 3

    def test_failed_status_exits(self, capsys):
        """Test that FAILED status causes sys.exit(1)."""
        mock_client = MagicMock()
        mock_client.get_harness.return_value = {"status": "FAILED"}

        with pytest.raises(SystemExit) as exc:
            wait_for_ready(mock_client, "test-abc123")
        assert exc.value.code == 1

    def test_timeout_raises(self):
        """Test that timeout raises TimeoutError."""
        mock_client = MagicMock()
        mock_client.get_harness.return_value = {"status": "CREATING"}

        with patch("agentcore_demo.time.sleep"), patch("agentcore_demo.time.time", side_effect=[0, 0, 200]):
            with pytest.raises(TimeoutError):
                wait_for_ready(mock_client, "test-abc123")


class TestInvokeHarness:
    """Test harness invocation and streaming."""

    def test_successful_invoke(self, capsys):
        """Test successful streaming invocation."""
        mock_client = MagicMock()
        mock_client.invoke_harness.return_value = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "AgentCore "}}},
                {"contentBlockDelta": {"delta": {"text": "is great."}}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        }

        response_text, latency = invoke_harness(mock_client, "arn:aws:bedrock-agentcore:us-east-1:123456789012:harness/test")

        assert response_text == "AgentCore is great."
        assert latency >= 0

        call_kwargs = mock_client.invoke_harness.call_args[1]
        assert call_kwargs["harnessArn"] == "arn:aws:bedrock-agentcore:us-east-1:123456789012:harness/test"
        assert call_kwargs["messages"][0]["content"][0]["text"] == TEST_PROMPT

    def test_stream_error_exits(self, capsys):
        """Test that a stream error causes sys.exit(1)."""
        mock_client = MagicMock()
        mock_client.invoke_harness.return_value = {
            "stream": [
                {"runtimeClientError": {"message": "Invocation failed"}},
            ]
        }

        with pytest.raises(SystemExit) as exc:
            invoke_harness(mock_client, "arn:aws:bedrock-agentcore:us-east-1:123456789012:harness/test")
        assert exc.value.code == 1

    def test_empty_stream(self, capsys):
        """Test invocation with empty stream returns empty string."""
        mock_client = MagicMock()
        mock_client.invoke_harness.return_value = {"stream": []}

        response_text, latency = invoke_harness(mock_client, "arn:aws:bedrock-agentcore:us-east-1:123456789012:harness/test")
        assert response_text == ""


class TestDeleteHarness:
    """Test harness deletion."""

    def test_successful_delete(self, capsys):
        """Test successful deletion."""
        mock_client = MagicMock()
        delete_harness(mock_client, "test-abc123")

        mock_client.delete_harness.assert_called_once_with(harnessId="test-abc123")
        captured = capsys.readouterr()
        assert "deleted" in captured.out

    def test_delete_failure_handled_gracefully(self, capsys):
        """Test that delete failure prints warning without raising."""
        mock_client = MagicMock()
        mock_client.delete_harness.side_effect = ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "Not found"}}, "delete_harness")

        delete_harness(mock_client, "test-abc123")

        captured = capsys.readouterr()
        assert "Could not delete" in captured.out
        assert "Manual cleanup" in captured.out


class TestConstants:
    """Test configuration constants."""

    def test_aws_region(self):
        assert AWS_REGION == "us-east-1"

    def test_model_id(self):
        assert HARNESS_MODEL_ID == "amazon.nova-pro-v1:0"

    def test_prompt_defined(self):
        assert isinstance(TEST_PROMPT, str)
        assert len(TEST_PROMPT) > 0
