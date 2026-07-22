#!/bin/bash
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
PREFIX="BedrockModelDemoHarness"

echo "Looking for harnesses with prefix '${PREFIX}' in ${REGION}..."

IDS=$(aws bedrock-agentcore-control list-harnesses \
  --region "${REGION}" \
  --query "harnesses[?starts_with(harnessName, '${PREFIX}')].harnessId" \
  --output text)

if [ -z "${IDS}" ]; then
  echo "No matching harnesses found."
  exit 0
fi

echo "${IDS}" | tr '\t' '\n' | while read -r id; do
  [ -z "${id}" ] && continue
  echo "Deleting harness: ${id}"
  aws bedrock-agentcore-control delete-harness \
    --harness-id "${id}" \
    --region "${REGION}"
done

echo "Done."
