"""
Triton Inference Server Client Wrapper
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class TritonClient:
    """Wrapper for Triton Inference Server client"""
    
    def __init__(self, url: str):
        self.url = url
        self.client = None
        self.gpu_available = False
        
        try:
            # Try to import tritonclient
            import tritonclient.grpc as grpcclient
            
            # Parse URL to extract host and port
            if "://" in url:
                url = url.split("://")[1]
            
            # Create client
            self.client = grpcclient.InferenceServerClient(url=url)
            
            # Check if server is live
            if self.client.is_server_live():
                logger.info(f"Connected to Triton server at {url}")
                self.gpu_available = self._check_gpu_availability()
            else:
                logger.warning(f"Triton server at {url} is not live")
                
        except ImportError:
            logger.warning("Triton client not available, using stub mode")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to connect to Triton: {e}")
            self.client = None
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available on Triton server"""
        try:
            if not self.client:
                return False
            
            # Get server metadata
            metadata = self.client.get_server_metadata()
            
            # Check for GPU in extensions
            if hasattr(metadata, 'extensions'):
                for ext in metadata.extensions:
                    if 'gpu' in ext.lower():
                        return True
            
            return False
        except Exception:
            return False
    
    def is_healthy(self) -> bool:
        """Check if Triton server is healthy"""
        try:
            if not self.client:
                return False
            return self.client.is_server_ready()
        except Exception:
            return False
    
    def has_gpu(self) -> bool:
        """Check if GPU is available"""
        return self.gpu_available
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            if not self.client:
                return []
            
            # Get model repository index
            models = self.client.get_model_repository_index()
            return [model.name for model in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information"""
        try:
            if not self.client:
                return {"error": "Client not connected"}
            
            # Get model metadata
            metadata = self.client.get_model_metadata(model_name)
            
            return {
                "name": metadata.name,
                "versions": metadata.versions,
                "platform": metadata.platform,
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": inp.shape
                    }
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": out.shape
                    }
                    for out in metadata.outputs
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: List[str],
        model_version: str = ""
    ) -> Dict[str, Any]:
        """Run inference on model"""
        start_time = time.time()
        
        try:
            if not self.client:
                # Stub mode - return mock results
                return self._stub_infer(model_name, inputs, outputs)
            
            # Import required classes
            import tritonclient.grpc as grpcclient
            
            # Prepare inputs
            triton_inputs = []
            for name, data in inputs.items():
                triton_input = grpcclient.InferInput(name, data.shape, "FP32")
                triton_input.set_data_from_numpy(data)
                triton_inputs.append(triton_input)
            
            # Prepare outputs
            triton_outputs = []
            for name in outputs:
                triton_outputs.append(grpcclient.InferRequestedOutput(name))
            
            # Run inference
            response = self.client.infer(
                model_name=model_name,
                inputs=triton_inputs,
                outputs=triton_outputs,
                model_version=model_version
            )
            
            # Parse results
            results = {}
            for output_name in outputs:
                results[output_name] = response.as_numpy(output_name)
            
            # Add timing
            results["inference_time"] = (time.time() - start_time) * 1000  # ms
            
            return results
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self._stub_infer(model_name, inputs, outputs)
    
    def _stub_infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        outputs: List[str]
    ) -> Dict[str, Any]:
        """Stub inference for when Triton is not available"""
        results = {}
        
        # Generate mock predictions
        for output in outputs:
            if output == "predictions":
                # Generate random probabilities
                num_classes = 8
                probs = np.random.random(num_classes)
                probs = probs / probs.sum()  # Normalize
                results[output] = probs.reshape(1, -1)
            elif output == "uncertainty":
                # Generate random uncertainty
                results[output] = np.array([np.random.random() * 0.5])
            else:
                results[output] = np.array([0.0])
        
        results["inference_time"] = np.random.random() * 100  # Random time 0-100ms
        
        return results
    
    def close(self):
        """Close client connection"""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass