"""Unit tests for AWS Bedrock model testing demonstrator."""

import json
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from bedrock_demo import (
    AWS_REGION,
    EMBEDDING_MODEL,
    EMBEDDING_TEST_INPUT,
    GENERATION_MODELS,
    MODEL_CATALOG,
    TEST_PROMPT,
    ResultsCollector,
    handle_error,
    run_nova_embeddings,
    run_nova_generation_models,
)


class TestResultsCollector:
    """Test the ResultsCollector class."""

    def test_initialization(self):
        """Test ResultsCollector initializes correctly."""
        results = ResultsCollector()
        assert results.successful == []
        assert results.failed == []
        assert results.total_time == 0.0

    def test_add_success(self):
        """Test adding successful results."""
        results = ResultsCollector()
        results.add_success("test-model-1", 1.5)
        results.add_success("test-model-2", 2.0)

        assert len(results.successful) == 2
        assert "test-model-1" in results.successful
        assert "test-model-2" in results.successful
        assert results.total_time == 3.5

    def test_add_failure(self):
        """Test adding failed results."""
        results = ResultsCollector()
        results.add_failure("test-model-1", "Access denied")
        results.add_failure("test-model-2", "Model not found")

        assert len(results.failed) == 2
        assert results.failed[0]["model"] == "test-model-1"
        assert results.failed[0]["error"] == "Access denied"
        assert results.failed[1]["model"] == "test-model-2"
        assert results.failed[1]["error"] == "Model not found"


class TestErrorHandling:
    """Test error handling functions."""

    def test_handle_access_denied_exception(self):
        """Test handling of AccessDeniedException."""
        error = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}}, "converse")
        result = handle_error(error, "test-model")
        assert "AccessDeniedException" in result
        assert "Enable model access" in result

    def test_handle_throttling_exception(self):
        """Test handling of ThrottlingException."""
        error = ClientError({"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "converse")
        result = handle_error(error, "test-model")
        assert "Rate limit exceeded" in result

    def test_handle_validation_exception(self):
        """Test handling of ValidationException."""
        error = ClientError({"Error": {"Code": "ValidationException", "Message": "Invalid parameter"}}, "converse")
        result = handle_error(error, "test-model")
        assert "ValidationException" in result

    def test_handle_resource_not_found_exception(self):
        """Test handling of ResourceNotFoundException."""
        error = ClientError({"Error": {"Code": "ResourceNotFoundException", "Message": "Model not found"}}, "converse")
        result = handle_error(error, "test-model")
        assert "ResourceNotFoundException" in result

    def test_handle_generic_exception(self):
        """Test handling of generic exceptions."""
        error = Exception("Something went wrong")
        result = handle_error(error, "test-model")
        assert "Unexpected error" in result
        assert "Something went wrong" in result


class TestNovaGenerationModels:
    """Test Nova generation model testing function."""

    def test_successful_nova_generation(self, capsys):
        """Test successful Nova generation model calls."""
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = {
            "output": {"message": {"content": [{"text": "AWS Bedrock is a fully managed service for foundation models."}]}},
            "usage": {"inputTokens": 10, "outputTokens": 15},
        }

        results = ResultsCollector()
        run_nova_generation_models(results, mock_bedrock)

        assert len(results.successful) == len(GENERATION_MODELS)
        assert len(results.failed) == 0

        captured = capsys.readouterr()
        assert "TESTING NOVA GENERATION MODELS" in captured.out

    def test_nova_generation_access_denied(self, capsys):
        """Test Nova generation with access denied error."""
        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}}, "converse")

        results = ResultsCollector()
        run_nova_generation_models(results, mock_bedrock)

        assert len(results.failed) == len(GENERATION_MODELS)
        captured = capsys.readouterr()
        assert "AccessDeniedException" in captured.out

    @patch("bedrock_demo.get_bedrock_client")
    def test_nova_generation_client_initialization_failure(self, mock_get_client, capsys):
        """Test Nova generation with client initialization failure."""
        mock_get_client.side_effect = Exception("Failed to initialize client")

        results = ResultsCollector()
        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = Exception("Failed to initialize client")
        run_nova_generation_models(results, mock_bedrock)

        assert len(results.failed) == len(GENERATION_MODELS)
        captured = capsys.readouterr()
        assert "Unexpected error" in captured.out


class TestNovaEmbeddings:
    """Test Nova embeddings model testing function."""

    def test_successful_embeddings(self, capsys):
        """Test successful embeddings generation."""
        mock_bedrock = MagicMock()
        mock_response = MagicMock()
        mock_response.__getitem__.return_value.read.return_value = json.dumps({"embedding": [0.1] * 1024}).encode("utf-8")
        mock_bedrock.invoke_model.return_value = mock_response

        results = ResultsCollector()
        run_nova_embeddings(results, mock_bedrock)

        assert len(results.successful) == 1
        assert EMBEDDING_MODEL in results.successful

        captured = capsys.readouterr()
        assert "TESTING EMBEDDINGS MODEL" in captured.out
        assert "Embedding generated successfully" in captured.out
        assert "Dimensions: 1024" in captured.out

    def test_successful_embeddings_plural_key(self, capsys):
        """Test successful embeddings with plural 'embeddings' response key."""
        mock_bedrock = MagicMock()
        mock_response = MagicMock()
        mock_response.__getitem__.return_value.read.return_value = json.dumps({"embeddings": [0.1] * 512}).encode("utf-8")
        mock_bedrock.invoke_model.return_value = mock_response

        results = ResultsCollector()
        run_nova_embeddings(results, mock_bedrock)

        assert len(results.successful) == 1
        captured = capsys.readouterr()
        assert "Dimensions: 512" in captured.out

    def test_embeddings_access_denied(self, capsys):
        """Test embeddings with access denied error."""
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}}, "invoke_model")

        results = ResultsCollector()
        run_nova_embeddings(results, mock_bedrock)

        assert len(results.failed) == 1
        captured = capsys.readouterr()
        assert "AccessDeniedException" in captured.out

    def test_embeddings_unexpected_response(self, capsys):
        """Test embeddings with unexpected response format."""
        mock_bedrock = MagicMock()
        mock_response = MagicMock()
        mock_response.__getitem__.return_value.read.return_value = json.dumps({"unexpected_field": "value"}).encode("utf-8")
        mock_bedrock.invoke_model.return_value = mock_response

        results = ResultsCollector()
        run_nova_embeddings(results, mock_bedrock)

        assert len(results.failed) == 1
        captured = capsys.readouterr()
        assert "no embedding found" in captured.out


class TestModelCatalog:
    """Test model catalog configuration."""

    def test_model_catalog_completeness(self):
        """Test that model catalog contains all expected models."""
        expected_models = [
            "amazon.nova-micro-v1:0",
            "amazon.nova-lite-v1:0",
            "amazon.nova-pro-v1:0",
            "us.amazon.nova-2-lite-v1:0",
            "amazon.nova-2-multimodal-embeddings-v1:0",
        ]

        for model_id in expected_models:
            assert model_id in MODEL_CATALOG
            assert "name" in MODEL_CATALOG[model_id]
            assert "type" in MODEL_CATALOG[model_id]
            assert "context" in MODEL_CATALOG[model_id]
            assert "description" in MODEL_CATALOG[model_id]

    def test_model_catalog_excludes_removed_models(self):
        """Test that removed models are not in the catalog."""
        assert "amazon.nova-premier-v1:0" not in MODEL_CATALOG
        assert "amazon.nova-2-lite-v1:0" not in MODEL_CATALOG
        assert "openai.gpt-5.4" not in MODEL_CATALOG
        assert "openai.gpt-5.5" not in MODEL_CATALOG
        assert "openai.gpt-5.6" not in MODEL_CATALOG

    def test_model_catalog_structure(self):
        """Test that each model catalog entry has correct structure."""
        for model_id, info in MODEL_CATALOG.items():
            assert isinstance(info["name"], str)
            assert isinstance(info["type"], str)
            assert isinstance(info["context"], str)
            assert isinstance(info["description"], str)
            assert len(info["name"]) > 0
            assert len(info["description"]) > 0

    def test_generation_models_derived_from_catalog(self):
        """Test that GENERATION_MODELS are derived from catalog and exclude embeddings."""
        assert "amazon.nova-2-multimodal-embeddings-v1:0" not in GENERATION_MODELS
        assert all(MODEL_CATALOG[m]["type"] != "Embeddings" for m in GENERATION_MODELS)

    def test_embedding_model_derived_from_catalog(self):
        """Test that EMBEDDING_MODEL is the sole embeddings entry in catalog."""
        assert EMBEDDING_MODEL == "amazon.nova-2-multimodal-embeddings-v1:0"
        assert MODEL_CATALOG[EMBEDDING_MODEL]["type"] == "Embeddings"


class TestConstants:
    """Test configuration constants."""

    def test_aws_region(self):
        """Test AWS region is set correctly."""
        assert AWS_REGION == "us-east-1"

    def test_test_prompt(self):
        """Test that test prompt is defined."""
        assert isinstance(TEST_PROMPT, str)
        assert len(TEST_PROMPT) > 0

    def test_embedding_test_input(self):
        """Test that embedding test input is defined."""
        assert isinstance(EMBEDDING_TEST_INPUT, str)
        assert len(EMBEDDING_TEST_INPUT) > 0
