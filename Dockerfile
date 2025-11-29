FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your script and any dependencies
COPY requirements.txt .
COPY ./src/get_data_from_previous_years.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the script
CMD ["python", "get_data_from_previous_years.py"]