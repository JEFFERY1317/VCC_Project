import requests
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine
import os
import random

# Get database connection parameters from environment variables
db_user = os.environ.get('DB_USER', 'mluser')
db_password = os.environ.get('DB_PASSWORD', 'password')
db_host = os.environ.get('DB_HOST', 'database')
db_name = os.environ.get('DB_NAME', 'housing_data')

# Connect to database to get real data for testing
connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
print(f"Connecting to database: {connection_string}")

# Wait a bit for everything to be ready
time.sleep(20)

engine = create_engine(connection_string)
test_data = pd.read_sql('SELECT * FROM housing LIMIT 100', engine).drop('target', axis=1)

# Convert to list of dictionaries for API calls
test_records = test_data.to_dict(orient='records')

# API endpoint
load_balancer_url = 'http://load-balancer'

# Function to make a single prediction
def make_prediction(record):
    url = f'{load_balancer_url}/predict'
    response = requests.post(url, json=record)
    return response.json()

# Function to run batch testing with different concurrency levels
def run_load_test(concurrency, num_requests):
    start_time = time.time()
    response_times = []
    container_hits = {}
    
    # Function to make request and record time
    def timed_request(record):
        start = time.time()
        try:
            result = make_prediction(record)
            end = time.time()
            response_times.append(end - start)
            
            # Track which container processed the request
            container_id = result.get('container_id', 'unknown')
            container_hits[container_id] = container_hits.get(container_id, 0) + 1
            
            return result
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    # Run concurrent requests
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Cycle through test records if needed
        test_data_extended = [test_records[i % len(test_records)] for i in range(num_requests)]
        results = list(executor.map(timed_request, test_data_extended))
    
    # Filter out None results (from errors)
    results = [r for r in results if r is not None]
    
    # Calculate metrics
    total_time = time.time() - start_time
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        p95_response_time = np.percentile(response_times, 95)
    else:
        avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
    throughput = len(results) / total_time if total_time > 0 else 0
    
    return {
        'concurrency': concurrency,
        'total_requests': num_requests,
        'successful_requests': len(results),
        'total_time_seconds': total_time,
        'avg_response_time_seconds': avg_response_time,
        'min_response_time_seconds': min_response_time,
        'max_response_time_seconds': max_response_time,
        'p95_response_time_seconds': p95_response_time,
        'throughput_rps': throughput,
        'container_hits': container_hits,
        'response_times': response_times
    }

# Test with different concurrency levels
concurrency_levels = [1, 2, 4, 8, 16, 32, 64]
results = []

for concurrency in concurrency_levels:
    print(f"Testing with concurrency level: {concurrency}")
    result = run_load_test(concurrency, 200)  # 200 requests per concurrency level
    results.append(result)
    print(f"Average response time: {result['avg_response_time_seconds']:.4f}s")
    print(f"P95 response time: {result['p95_response_time_seconds']:.4f}s")
    print(f"Throughput: {result['throughput_rps']:.2f} requests/second")
    print(f"Container hits: {result['container_hits']}")
    print("-" * 50)
    
    # Wait between tests to let system stabilize
    time.sleep(5)

# Plot results
plt.figure(figsize=(12, 10))

# Plot 1: Throughput vs Concurrency
plt.subplot(2, 2, 1)
plt.plot([r['concurrency'] for r in results], [r['throughput_rps'] for r in results], 'o-', linewidth=2)
plt.xlabel('Concurrency Level')
plt.ylabel('Throughput (requests/second)')
plt.title('Throughput vs Concurrency')
plt.grid(True)

# Plot 2: Average Response Time vs Concurrency
plt.subplot(2, 2, 2)
plt.plot([r['concurrency'] for r in results], [r['avg_response_time_seconds'] * 1000 for r in results], 'o-', linewidth=2)
plt.xlabel('Concurrency Level')
plt.ylabel('Average Response Time (ms)')
plt.title('Response Time vs Concurrency')
plt.grid(True)

# Plot 3: P95 Response Time vs Concurrency
plt.subplot(2, 2, 3)
plt.plot([r['concurrency'] for r in results], [r['p95_response_time_seconds'] * 1000 for r in results], 'o-', linewidth=2)
plt.xlabel('Concurrency Level')
plt.ylabel('P95 Response Time (ms)')
plt.title('P95 Response Time vs Concurrency')
plt.grid(True)

# Plot 4: Container Hits
plt.subplot(2, 2, 4)
containers = set()
for r in results:
    containers.update(r['container_hits'].keys())

container_names = sorted(list(containers))
container_data = {container: [] for container in container_names}

for r in results:
    for container in container_names:
        container_data[container].append(r['container_hits'].get(container, 0))

bottom = np.zeros(len(results))
for container in container_names:
    plt.bar([str(r['concurrency']) for r in results], container_data[container], bottom=bottom, label=container)
    bottom += np.array(container_data[container])

plt.xlabel('Concurrency Level')
plt.ylabel('Number of Requests')
plt.title('Container Load Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('/app/load_test_results.png')
print("Load test results saved as /app/load_test_results.png")

# Save detailed results to CSV for further analysis
detailed_results = []
for r in results:
    for rt in r['response_times']:
        detailed_results.append({
            'concurrency': r['concurrency'],
            'response_time_seconds': rt
        })

pd.DataFrame(detailed_results).to_csv('/app/detailed_load_test_results.csv', index=False)
print("Detailed results saved to /app/detailed_load_test_results.csv")

# Save summary results
summary_results = [{
    'concurrency': r['concurrency'],
    'throughput_rps': r['throughput_rps'],
    'avg_response_time_ms': r['avg_response_time_seconds'] * 1000,
    'p95_response_time_ms': r['p95_response_time_seconds'] * 1000,
    'min_response_time_ms': r['min_response_time_seconds'] * 1000,
    'max_response_time_ms': r['max_response_time_seconds'] * 1000
} for r in results]

pd.DataFrame(summary_results).to_csv('/app/summary_load_test_results.csv', index=False)
print("Summary results saved to /app/summary_load_test_results.csv")

# Also save container hit distribution
container_hit_results = []
for r in results:
    row = {'concurrency': r['concurrency']}
    for container, hits in r['container_hits'].items():
        row[container] = hits
    container_hit_results.append(row)

pd.DataFrame(container_hit_results).to_csv('/app/container_distribution_results.csv', index=False)
print("Container distribution results saved to /app/container_distribution_results.csv")
