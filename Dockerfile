# Base Image

FROM python:3.10

# WORKDIR
WORKDIR /app

#COPY
COPY . /app


# INSTALL DEPENDENCIES
RUN pip install --no-cache-dir -r requirements.txt

# PORT
EXPOSE 8000

# COMMAND
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
