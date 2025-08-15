/**
 * K6 Load Test for CXR (Chest X-Ray) Endpoint
 * Tests the imaging service under load
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 10 },  // Ramp up to 10 users
    { duration: '1m', target: 10 },   // Stay at 10 users
    { duration: '30s', target: 20 },  // Ramp up to 20 users
    { duration: '1m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    'http_req_duration': ['p(95)<2000'], // 95% of requests must complete below 2s
    'errors': ['rate<0.1'],               // Error rate must be below 10%
  },
};

// Test data
const IMAGING_URL = __ENV.IMAGING || 'http://localhost:8006';
const AUTH_TOKEN = __ENV.TOKEN || '';

// Sample CXR image (base64 encoded 1x1 pixel for testing)
const SAMPLE_IMAGE = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

export default function () {
  // Prepare request
  const params = {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  };
  
  if (AUTH_TOKEN) {
    params.headers['Authorization'] = `Bearer ${AUTH_TOKEN}`;
  }
  
  // Create form data
  const formData = {
    file: http.file(SAMPLE_IMAGE, 'test_cxr.png', 'image/png'),
    modality: 'CXR',
    return_uncertainty: 'true',
  };
  
  // Test 1: Health check
  const healthRes = http.get(`${IMAGING_URL}/v1/triton/health`);
  check(healthRes, {
    'health check status is 200': (r) => r.status === 200,
    'health check returns healthy': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.healthy === true;
      } catch (e) {
        return false;
      }
    },
  });
  
  // Test 2: CXR inference
  const inferRes = http.post(
    `${IMAGING_URL}/v1/imaging/infer`,
    formData,
    params
  );
  
  const inferSuccess = check(inferRes, {
    'inference status is 200': (r) => r.status === 200,
    'inference returns predictions': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.predictions !== undefined;
      } catch (e) {
        return false;
      }
    },
    'inference time is reasonable': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.inference_time_ms < 1000;
      } catch (e) {
        return false;
      }
    },
  });
  
  errorRate.add(!inferSuccess);
  
  // Test 3: Model listing
  const modelsRes = http.get(`${IMAGING_URL}/v1/models`, params);
  check(modelsRes, {
    'models list status is 200': (r) => r.status === 200,
    'models list returns array': (r) => {
      try {
        const body = JSON.parse(r.body);
        return Array.isArray(body.models);
      } catch (e) {
        return false;
      }
    },
  });
  
  // Sleep between iterations
  sleep(1);
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'cxr_load_test_results.json': JSON.stringify(data),
  };
}

function textSummary(data, options) {
  // Simple text summary
  const { metrics } = data;
  let summary = '\n=== CXR Load Test Results ===\n';
  
  if (metrics) {
    summary += `\nRequests: ${metrics.http_reqs?.values?.count || 0}\n`;
    summary += `Errors: ${metrics.errors?.values?.rate || 0}%\n`;
    summary += `Avg Duration: ${metrics.http_req_duration?.values?.avg || 0}ms\n`;
    summary += `P95 Duration: ${metrics.http_req_duration?.values['p(95)'] || 0}ms\n`;
  }
  
  return summary;
}