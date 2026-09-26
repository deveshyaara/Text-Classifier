FROM python:3.11

WORKDIR /code

# Install dependencies first for better caching
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application files (including the model)
COPY . .

# Run the FastAPI server on Hugging Face's default port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
