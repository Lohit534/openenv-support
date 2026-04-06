FROM python:3.10-slim

WORKDIR /app

# Ensure lightweight build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Inference Environment Variables
ENV API_BASE_URL="https://api-inference.huggingface.co/v1/"
ENV MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

CMD ["uvicorn", "env.environment:app", "--host", "0.0.0.0", "--port", "7860"]
