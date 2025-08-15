#!/usr/bin/env python3
"""
Med-AGI Smoke Test Suite
Pre-pilot validation tests for all services
"""

import argparse
import sys
import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    import requests
except ImportError:
    print("Error: requests library not installed")
    print("Install with: pip install requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result container"""
    name: str
    passed: bool
    message: str
    duration_ms: float
    details: Optional[Dict] = None


class SmokeTests:
    """Smoke test suite for Med-AGI services"""
    
    def __init__(self, args):
        self.args = args
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
        # Service endpoints
        self.services = {
            'imaging': args.imaging,
            'ekg': args.ekg if hasattr(args, 'ekg') else None,
            'eval': args.eval,
            'anchor': args.anchor,
            'modelcards': args.modelcards,
            'ops': args.ops
        }
        
        # Remove None values
        self.services = {k: v for k, v in self.services.items() if v}
        
        # JWT token for authenticated endpoints
        self.token = args.token if hasattr(args, 'token') else None
        self.headers = {}
        if self.token:
            self.headers['Authorization'] = f'Bearer {self.token}'
    
    def run_test(self, name: str, test_func) -> TestResult:
        """Run a single test and capture result"""
        logger.info(f"Running test: {name}")
        start = time.time()
        
        try:
            passed, message, details = test_func()
            duration = (time.time() - start) * 1000
            
            result = TestResult(
                name=name,
                passed=passed,
                message=message,
                duration_ms=duration,
                details=details
            )
            
            if passed:
                logger.info(f"✓ {name}: {message} ({duration:.2f}ms)")
            else:
                logger.error(f"✗ {name}: {message}")
                
        except Exception as e:
            duration = (time.time() - start) * 1000
            result = TestResult(
                name=name,
                passed=False,
                message=f"Exception: {str(e)}",
                duration_ms=duration
            )
            logger.error(f"✗ {name}: Exception - {str(e)}")
        
        self.results.append(result)
        return result
    
    def test_service_health(self, service: str, endpoint: str) -> tuple:
        """Test service health endpoint"""
        try:
            url = f"{self.services[service]}{endpoint}"
            response = requests.get(url, timeout=5, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                return True, f"Service healthy", data
            else:
                return False, f"Status {response.status_code}", None
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection failed: {str(e)}", None
    
    def test_imaging_service(self):
        """Test imaging service endpoints"""
        if 'imaging' not in self.services:
            return
        
        # Test health
        self.run_test(
            "imaging_health",
            lambda: self.test_service_health('imaging', '/v1/triton/health')
        )
        
        # Test model listing
        def test_models():
            try:
                url = f"{self.services['imaging']}/v1/models"
                response = requests.get(url, timeout=5, headers=self.headers)
                
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    return True, f"Found {len(models)} models", {'models': models}
                else:
                    return False, f"Status {response.status_code}", None
            except Exception as e:
                return False, str(e), None
        
        self.run_test("imaging_models", test_models)
    
    def test_ekg_service(self):
        """Test EKG service endpoints"""
        if 'ekg' not in self.services:
            return
        
        # Test health
        self.run_test(
            "ekg_health",
            lambda: self.test_service_health('ekg', '/v1/ekg/health')
        )
        
        # Test inference endpoint
        def test_inference():
            try:
                url = f"{self.services['ekg']}/v1/ekg/infer"
                # Generate test data
                samples = [0.1] * 1000  # 1000 sample points
                
                response = requests.post(
                    url,
                    json={'samples': samples},
                    timeout=10,
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'probs' in data:
                        return True, "Inference successful", data
                    else:
                        return False, "Invalid response format", data
                else:
                    return False, f"Status {response.status_code}", None
                    
            except Exception as e:
                return False, str(e), None
        
        if not self.args.skip_inference:
            self.run_test("ekg_inference", test_inference)
    
    def test_eval_service(self):
        """Test evaluation service"""
        if 'eval' not in self.services:
            return
        
        # Test summary endpoint
        def test_summary():
            try:
                url = f"{self.services['eval']}/v1/eval/summary"
                response = requests.get(url, timeout=5, headers=self.headers)
                
                if response.status_code == 200:
                    return True, "Summary retrieved", response.json()
                else:
                    return False, f"Status {response.status_code}", None
                    
            except Exception as e:
                return False, str(e), None
        
        self.run_test("eval_summary", test_summary)
    
    def test_anchor_service(self):
        """Test anchor service"""
        if 'anchor' not in self.services:
            return
        
        # Test search endpoint
        def test_search():
            try:
                url = f"{self.services['anchor']}/v1/anchor_search"
                response = requests.get(
                    url,
                    params={'query': 'test'},
                    timeout=5,
                    headers=self.headers
                )
                
                if response.status_code in [200, 404]:
                    return True, f"Search endpoint responsive", None
                else:
                    return False, f"Status {response.status_code}", None
                    
            except Exception as e:
                return False, str(e), None
        
        self.run_test("anchor_search", test_search)
    
    def test_modelcards_service(self):
        """Test modelcards service"""
        if 'modelcards' not in self.services:
            return
        
        # Test list endpoint
        def test_list():
            try:
                url = f"{self.services['modelcards']}/v1/modelcards/list"
                response = requests.get(url, timeout=5, headers=self.headers)
                
                if response.status_code == 200:
                    cards = response.json()
                    count = len(cards) if isinstance(cards, list) else 0
                    return True, f"Found {count} model cards", None
                else:
                    return False, f"Status {response.status_code}", None
                    
            except Exception as e:
                return False, str(e), None
        
        self.run_test("modelcards_list", test_list)
    
    def test_ops_service(self):
        """Test operations service"""
        if 'ops' not in self.services:
            return
        
        # Test SIEM stats endpoint
        def test_siem():
            try:
                url = f"{self.services['ops']}/v1/ops/siem/stats"
                response = requests.get(url, timeout=5, headers=self.headers)
                
                if response.status_code == 200:
                    stats = response.json()
                    return True, "SIEM stats retrieved", stats
                else:
                    return False, f"Status {response.status_code}", None
                    
            except Exception as e:
                return False, str(e), None
        
        self.run_test("ops_siem_stats", test_siem)
    
    def test_cross_service_communication(self):
        """Test communication between services"""
        if len(self.services) < 2:
            return
        
        # Test if services can reach each other
        def test_connectivity():
            reachable = 0
            total = len(self.services)
            
            for service, url in self.services.items():
                try:
                    # Try basic health endpoint
                    response = requests.get(f"{url}/health", timeout=2)
                    if response.status_code < 500:
                        reachable += 1
                except:
                    pass
            
            if reachable == total:
                return True, f"All {total} services reachable", None
            else:
                return False, f"Only {reachable}/{total} services reachable", None
        
        self.run_test("cross_service_connectivity", test_connectivity)
    
    def run_all_tests(self):
        """Run all smoke tests"""
        logger.info("=" * 60)
        logger.info("Starting Med-AGI Smoke Tests")
        logger.info(f"Services: {', '.join(self.services.keys())}")
        logger.info("=" * 60)
        
        # Run service-specific tests
        self.test_imaging_service()
        self.test_ekg_service()
        self.test_eval_service()
        self.test_anchor_service()
        self.test_modelcards_service()
        self.test_ops_service()
        
        # Run cross-service tests
        self.test_cross_service_communication()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate test report"""
        total_duration = (time.time() - self.start_time) * 1000
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        logger.info("=" * 60)
        logger.info("Test Results Summary")
        logger.info("=" * 60)
        logger.info(f"Total tests: {len(self.results)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Duration: {total_duration:.2f}ms")
        
        if failed > 0:
            logger.info("\nFailed tests:")
            for result in self.results:
                if not result.passed:
                    logger.info(f"  - {result.name}: {result.message}")
        
        # Write JSON report
        if self.args.output:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': len(self.results),
                    'passed': passed,
                    'failed': failed,
                    'duration_ms': total_duration
                },
                'services': list(self.services.keys()),
                'results': [
                    {
                        'name': r.name,
                        'passed': r.passed,
                        'message': r.message,
                        'duration_ms': r.duration_ms,
                        'details': r.details
                    }
                    for r in self.results
                ]
            }
            
            with open(self.args.output, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"\nReport written to: {self.args.output}")
        
        # Exit with appropriate code
        if failed > 0:
            sys.exit(1)
        else:
            logger.info("\n✓ All smoke tests passed!")
            sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Med-AGI Smoke Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Service endpoints
    parser.add_argument('--imaging', required=True, help='Imaging service URL')
    parser.add_argument('--ekg', help='EKG service URL')
    parser.add_argument('--eval', required=True, help='Eval service URL')
    parser.add_argument('--anchor', required=True, help='Anchor service URL')
    parser.add_argument('--modelcards', required=True, help='ModelCards service URL')
    parser.add_argument('--ops', required=True, help='Ops service URL')
    
    # Options
    parser.add_argument('--token', help='JWT token for authentication')
    parser.add_argument('--output', help='Output JSON report file')
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip inference tests (faster)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    tester = SmokeTests(args)
    tester.run_all_tests()


if __name__ == '__main__':
    main()