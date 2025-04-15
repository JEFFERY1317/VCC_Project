import requests
import json
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import os
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice

# Get workspace
subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "<your-subscription-id>")
resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "house-rg")
workspace_name = os.environ.get("AZURE_WORKSPACE_NAME", "housing-ml-workspace")

ws = Workspace.get(
    name=workspace_name,
    subscription_id=subscription_id,
    resource_group=resource_group
)

# Get service
service_name = 'housing-price-service'
service = AciWebservice(ws, service_name)

# Prepare test data
test_data = {
    "data": [
        {
            "MedInc": 8.3252,
            "HouseAge": 41.0,
            "AveRooms": 6.984127,
            "AveBedrms": 1.023810,
            "Population": 322.0,
            "AveOccup": 2.555556,
            "Latitude": 37.88,
            "Longitude": -122.23
        },
        {
            "MedInc": 5.6431,
            "HouseAge": 21.0,
            "AveRooms": 6.238137,
            "AveBedrms": 0.971880,
            "Population": 2401.0,
            "AveOccup": 2.109842,
            "Latitude": 37.85,
            "Longitude": -122.25
        }
    ]
}

# Make a single request
def test_single_request():
    # Get service authentication keys
    key1, key2 = service.get_keys()
    
    # Set headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key1}'
    }
    
    # Send request
    start_time = time.time()
    response = requests.post(service.scoring_uri, json=test_data, headers=headers)
    latency = (time.time() - start_time) * 1000  # Convert to ms
    
    # Print response
    if response.status_code == 200:
        result = response.json()
        print(f"Predictions: {result['predictions']}")
        print(f"Service response time: {result['response_time_ms']:.2f} ms")
        print(f"End-to-end latency: {latency:.2f} ms")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

# Function to send a request and measure response time
def send_request():
    # Get service authentication keys
    key1, key2 = service.get_keys()
    
    # Set headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key1}'
    }
    
    # Send request
    start_time = time.time()
    response = requests.post(service.scoring_uri, json=test_data, headers=headers)
    latency = (time.time() - start_time) * 1000  # Convert to ms
    
    if response.status_code == 200:
        result = response.json()
        return {
            "success": True,
            "service_time": result['response_time_ms'],
            "total_time": latency
        }
    else:
        return {
            "success": False,
            "error_code": response.status_code
        }

# Load test function
def run_load_test(concurrency, num_requests):
    results = []
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]
        for future in futures:
            results.append(future.result())
    
    # Filter successful results
    success_results = [r for r in results if r.get("success", False)]
    
    # Calculate metrics
    if success_results:
        service_times = [r["service_time"] for r in success_results]
        total_times = [r["total_time"] for r in success_results]
        
        metrics = {
            "concurrency": concurrency,
            "num_requests": num_requests,
            "success_rate": len(success_results) / num_requests * 100,
            "avg_service_time": statistics.mean(service_times),
            "p95_service_time": np.percentile(service_times, 95),
            "avg_total_time": statistics.mean(total_times),
            "p95_total_time": np.percentile(total_times, 95)
        }
    else:
        metrics = {
            "concurrency": concurrency,
            "num_requests": num_requests,
            "success_rate": 0,
            "avg_service_time": 0,
            "p95_service_time": 0,
            "avg_total_time": 0,
            "p95_total_time": 0
        }
    
    return metrics

# Run single test
print("Testing single request...")
test_single_request()

# Run load test at different concurrency levels
print("\nRunning load tests...")

concurrency_levels = [1, 2, 4, 8]
num_requests = 50  # 50 requests per concurrency level
results = []

for concurrency in concurrency_levels:
    print(f"Testing with concurrency level: {concurrency}")
    metrics = run_load_test(concurrency, num_requests)
    results.append(metrics)
    print(f"Success rate: {metrics['success_rate']:.2f}%")
    print(f"Average service time: {metrics['avg_service_time']:.2f} ms")
    print(f"P95 service time: {metrics['p95_service_time']:.2f} ms")
    print(f"Average total time: {metrics['avg_total_time']:.2f} ms")
    print(f"P95 total time: {metrics['p95_total_time']:.2f} ms")
    print("-" * 50)

# Create plots
plt.figure(figsize=(12, 8))

# Plot 1: Average Response Time vs Concurrency
plt.subplot(2, 2, 1)
plt.plot([r["concurrency"] for r in results], [r["avg_service_time"] for r in results], 'o-', label='Service Time')
plt.plot([r["concurrency"] for r in results], [r["avg_total_time"] for r in results], 's-', label='Total Time')
plt.xlabel('Concurrency Level')
plt.ylabel('Average Response Time (ms)')
plt.title('Response Time vs Concurrency')
plt.legend()
plt.grid(True)

# Plot 2: P95 Response Time vs Concurrency
plt.subplot(2, 2, 2)
plt.plot([r["concurrency"] for r in results], [r["p95_service_time"] for r in results], 'o-', label='Service Time')
plt.plot([r["concurrency"] for r in results], [r["p95_total_time"] for r in results], 's-', label='Total Time')
plt.xlabel('Concurrency Level')
plt.ylabel('P95 Response Time (ms)')
plt.title('P95 Response Time vs Concurrency')
plt.legend()
plt.grid(True)

# Plot 3: Success Rate vs Concurrency
plt.subplot(2, 2, 3)
plt.plot([r["concurrency"] for r in results], [r["success_rate"] for r in results], 'o-')
plt.xlabel('Concurrency Level')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate vs Concurrency')
plt.grid(True)

# Calculate throughput (requests per second)
for r in results:
    r["throughput"] = (r["num_requests"] * (r["success_rate"] / 100)) / (r["avg_total_time"] * r["num_requests"] / 1000)

# Plot 4: Throughput vs Concurrency
plt.subplot(2, 2, 4)
plt.plot([r["concurrency"] for r in results], [r["throughput"] for r in results], 'o-')
plt.xlabel('Concurrency Level')
plt.ylabel('Throughput (req/sec)')
plt.title('Throughput vs Concurrency')
plt.grid(True)

plt.tight_layout()
plt.savefig('azure_ml_load_test_results.png')
print("Load test results plot saved as azure_ml_load_test_results.png")

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('azure_ml_load_test_results.csv', index=False)
print("Load test results saved to azure_ml_load_test_results.csv")
