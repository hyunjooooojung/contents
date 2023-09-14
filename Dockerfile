FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy your FastAPI application code to the container
COPY . .

# Install any necessary dependencies
RUN pip install -r requirements.txt

# # Expose the port that your FastAPI app will listen on
EXPOSE 80

CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:80", "--workers", "4"]

