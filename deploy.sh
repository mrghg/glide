#!/usr/bin/env bash
set -euo pipefail

# Required configuration. Override with env vars if desired.
PROJECT_ID="${PROJECT_ID:-your-gcp-project-id}"
REGION="${REGION:-us-central1}"
REPOSITORY="${REPOSITORY:-lpdm}"
IMAGE_NAME="${IMAGE_NAME:-lpdm-backward}"
SERVICE_NAME="${SERVICE_NAME:-lpdm-backward}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${TAG}"

if [[ "${PROJECT_ID}" == "your-gcp-project-id" ]]; then
  echo "ERROR: Set PROJECT_ID before running this script."
  echo "Example: PROJECT_ID=my-project ./deploy.sh"
  exit 1
fi

echo "==> Authenticating with Google Cloud"
gcloud auth login
gcloud config set project "${PROJECT_ID}"

echo "==> Enabling required APIs"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

echo "==> Creating Artifact Registry repo if needed"
if ! gcloud artifacts repositories describe "${REPOSITORY}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="LPDM container images"
fi

echo "==> Configuring Docker auth for Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "==> Building and pushing image with Cloud Build"
gcloud builds submit --tag "${IMAGE_URI}" .

echo "==> Deploying to Cloud Run with NVIDIA L4 GPU"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --execution-environment=gen2 \
  --gpu=1 \
  --gpu-type=nvidia-l4 \
  --cpu=8 \
  --memory=32Gi \
  --timeout=3600 \
  --concurrency=1 \
  --max-instances=5 \
  --no-allow-unauthenticated

echo "==> Deployment complete"
gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --format='value(status.url)'
