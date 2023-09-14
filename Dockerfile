FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy your FastAPI application code to the container
COPY . .

# Install any necessary dependencies
RUN pip install -r requirements.txt

# Expose the port that your FastAPI app will listen on
EXPOSE 8001

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
