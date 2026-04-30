# Use Python 3.10 (important for ML libs)
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (HuggingFace uses 7860)
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]