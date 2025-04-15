#!/bin/bash

echo "Building and starting containerized ML model deployment..."

# Create needed directories
mkdir -p api/data api/models results

# Build all containers
echo "Building containers..."
docker-compose build

# Start the core services
echo "Starting database, data setup, and model training..."
docker-compose up -d database
docker-compose up data-setup
docker-compose up model-training

# Start the API containers
echo "Starting API containers..."
docker-compose up -d api1 api2 api3 api4

# Start the load balancer and monitoring
echo "Starting load balancer and monitoring service..."
docker-compose up -d load-balancer monitoring

# Wait for all services to be ready
echo "Waiting for all services to be ready..."
sleep 30

# Print the status of all services
echo "Current status of services:"
docker-compose ps

# Print access information
echo ""
echo "Access information:"
echo "- Load Balancer (API): http://localhost"
echo "- Monitoring Dashboard: http://localhost:8080"
echo ""

# Ask if user wants to run load testing
read -p "Do you want to run load testing now? (y/n): " run_test
if [[ $run_test == "y" || $run_test == "Y" ]]; then
    echo "Running load testing..."
    docker-compose up load-testing
    
    echo "Load testing complete. Results are available in the ./results directory."
fi

echo ""
echo "To stop all services: docker-compose down"
echo "To stop a specific service: docker-compose stop [service_name]"
echo "To view logs: docker-compose logs [service_name]"
