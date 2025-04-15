# load_test.py
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.datasets import fetch_california_housing

# Get test data from California housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
test_records = df.head(100).to_dict('records')

# Replace with your VM’s external IP
api_url = "http://34.47.152.52:8080/predict"

# Function to make a single prediction
def make_prediction(record):
    start = time.time()
    response = requests.post(api_url, json=record)
    end = time.time()
    response_time = end - start
    if response.status_code == 200:
        return {
            'success': True,
            'response_time': response_time,
            'result': response.json()
        }
    else:
        return {
            'success': False,
            'response_time': response_time,
            'error': response.text
        }

# Test with different concurrency levels
concurrency_levels = [1, 2, 4, 8, 16, 32]
results = []

for concurrency in concurrency_levels:
    print(f"Testing with concurrency level: {concurrency}")
    start_time = time.time()
    response_times = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        test_data = [test_records[i % len(test_records)] for i in range(100)]
        result_list = list(executor.map(make_prediction, test_data))
    successful = [r for r in result_list if r['success']]
    if successful:
        response_times = [r['response_time'] for r in successful]
    total_time = time.time() - start_time
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    throughput = len(successful) / total_time if total_time > 0 else 0
    results.append({
        'concurrency': concurrency,
        'avg_response_time_ms': avg_response_time * 1000,
        'throughput_rps': throughput,
        'success_rate': len(successful) / 100
    })
    print(f"Average response time: {avg_response_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.2f} requests/second")
    print(f"Success rate: {len(successful)}%")
    print("-" * 30)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot([r['concurrency'] for r in results], [r['throughput_rps'] for r in results], 'o-')
plt.title('Throughput vs Concurrency')
plt.xlabel('Concurrency')
plt.ylabel('Throughput (req/sec)')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot([r['concurrency'] for r in results], [r['avg_response_time_ms'] for r in results], 'o-')
plt.title('Response Time vs Concurrency')
plt.xlabel('Concurrency')
plt.ylabel('Avg Response Time (ms)')
plt.grid(True)
plt.tight_layout()
plt.savefig('cloud_performance.png')
plt.show()
pd.DataFrame(results).to_csv('cloud_results.csv', index=False)
