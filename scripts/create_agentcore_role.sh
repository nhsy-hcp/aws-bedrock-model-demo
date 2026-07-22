#!/bin/bash
set -euo pipefail

ROLE_NAME="${AGENTCORE_ROLE_NAME:-bedrock-agentcore-demo-role}"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Creating AgentCore execution role: ${ROLE_NAME}"
echo "Account: ${ACCOUNT_ID} | Region: ${REGION}"

TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Service": "bedrock-agentcore.amazonaws.com"
    },
    "Action": "sts:AssumeRole",
    "Condition": {
      "StringEquals": {
        "aws:SourceAccount": "${ACCOUNT_ID}"
      }
    }
  }]
}
EOF
)

PERMISSIONS_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:${REGION}::foundation-model/amazon.nova-pro-v1:0"
    },
    {
      "Sid": "AgentCoreMemory",
      "Effect": "Allow",
      "Action": "bedrock-agentcore:*",
      "Resource": "arn:aws:bedrock-agentcore:${REGION}:${ACCOUNT_ID}:memory/*"
    },
    {
      "Sid": "Logs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/bedrock-agentcore/*"
    }
  ]
}
EOF
)

if aws iam get-role --role-name "${ROLE_NAME}" > /dev/null 2>&1; then
  echo "Role '${ROLE_NAME}' already exists — updating trust policy."
  aws iam update-assume-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-document "${TRUST_POLICY}"
else
  echo "Creating role '${ROLE_NAME}'..."
  aws iam create-role \
    --role-name "${ROLE_NAME}" \
    --assume-role-policy-document "${TRUST_POLICY}" \
    --description "Execution role for Bedrock AgentCore demo harness" \
    --output text > /dev/null
fi

echo "Putting inline permissions policy..."
aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "AgentCoreDemoPermissions" \
  --policy-document "${PERMISSIONS_POLICY}"

ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query "Role.Arn" --output text)

echo ""
echo "Done. Role ARN:"
echo "  ${ROLE_ARN}"
echo ""
echo "Export for use with the demo:"
echo "  export AGENTCORE_EXECUTION_ROLE_ARN=${ROLE_ARN}"
