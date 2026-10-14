import time
import random
import os

from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentel_sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentel_sdk.metrics import MeterProvider
from opentel_sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentel_sdk.metrics import ObservableGauge

# --- 1. OpenTelemetry Setup ---
# Resource for identifying our service
resource = Resource.create({
    "service.name": "llm-inference-gateway",
    "service.version": "1.0.0",
    "environment": "development"
})

# Tracer Provider setup
trace_provider = TracerProvider(resource=resource)
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
trace_provider.add_span_processor(span_processor)
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)

# Metric Provider setup
# We'll use a PeriodicExportingMetricReader with a ConsoleMetricExporter
metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
metric_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(metric_provider)
meter = metrics.get_meter(__name__)

# Define metrics
inference_counter = meter.create_counter(
    "llm.inference.calls",
    description="Total number of LLM inference calls",
    unit="1"
)
inference_latency_histogram = meter.create_histogram(
    "llm.inference.latency",
    description="Latency of LLM inference calls",
    unit="ms"
)

# Global variable to hold current simulated GPU utilization
simulated_gpu_utilization = 0.0 

def get_simulated_gpu_utilization_callback(options):
    """Callback for the ObservableGauge to report current simulated GPU utilization."""
    yield ObservableGauge.Observation(simulated_gpu_utilization, {"device": "gpu:0"})

meter.create_observable_gauge(
    "gpu.utilization",
    callbacks=[get_simulated_gpu_utilization_callback],
    description="Simulated GPU utilization percentage",
    unit="%"
)


# --- 2. Simulated LLM Backends ---
def _call_llm_a(prompt: str):
    """Simulates a faster, cheaper LLM backend."""
    with tracer.start_as_current_span("llm_backend_a_inference") as span:
        span.set_attribute("llm.model_name", "LLM-A-Fast")
        span.set_attribute("llm.model_cost_per_token", 0.0001) # Simulated cost per token
        
        inference_time_ms = random.uniform(50, 200) # Simulate 50-200ms inference
        time.sleep(inference_time_ms / 1000)
        
        response = f"Response from LLM-A for '{prompt[:20]}...'"
        cost = len(prompt) * 0.0001 + len(response) * 0.00005 # Example cost calculation
        
        span.set_attribute("llm.inference_time_ms", inference_time_ms)
        span.set_attribute("llm.simulated_cost", cost)
        span.set_attribute("llm.response_length", len(response))
        
        return response, cost

def _call_llm_b(prompt: str):
    """Simulates a slower, more expensive LLM backend for complex tasks."""
    with tracer.start_as_current_span("llm_backend_b_inference") as span:
        span.set_attribute("llm.model_name", "LLM-B-Complex")
        span.set_attribute("llm.model_cost_per_token", 0.0005) # Simulated cost per token
        
        inference_time_ms = random.uniform(200, 800) # Simulate 200-800ms inference
        time.sleep(inference_time_ms / 1000)
        
        response = f"Detailed response from LLM-B for '{prompt[:20]}...'"
        cost = len(prompt) * 0.0005 + len(response) * 0.0002 # Example cost calculation
        
        span.set_attribute("llm.inference_time_ms", inference_time_ms)
        span.set_attribute("llm.simulated_cost", cost)
        span.set_attribute("llm.response_length", len(response))
        
        return response, cost

# --- 3. LLM Gateway Function ---
def llm_gateway(prompt: str):
    """
    Simulates an LLM gateway that routes requests based on prompt length
    and captures OTel traces and metrics, including simulated GPU usage.
    """
    global simulated_gpu_utilization # Access global variable to update GPU metric

    with tracer.start_as_current_span("llm_gateway_inference") as span:
        start_time = time.perf_counter()
        
        span.set_attribute("llm.prompt_length", len(prompt))
        span.set_attribute("llm.request_id", str(random.randint(1000, 9999)))

        chosen_model = ""
        response = ""
        total_cost = 0.0
        
        # Simple routing logic: shorter prompts go to LLM-A, longer to LLM-B
        if len(prompt) < 50:
            chosen_model = "LLM-A-Fast"
            response, total_cost = _call_llm_a(prompt)
            # Simulate lower GPU usage for simpler models
            simulated_gpu_utilization = random.uniform(10.0, 40.0) # OTel-native GPU profiler (simulated)
        else:
            chosen_model = "LLM-B-Complex"
            response, total_cost = _call_llm_b(prompt)
            # Simulate higher GPU usage for complex models
            simulated_gpu_utilization = random.uniform(50.0, 90.0) # OTel-native GPU profiler (simulated)
        
        end_time = time.perf_counter()
        inference_duration_ms = (end_time - start_time) * 1000

        # Increment inference counter and record latency using OTel metrics
        inference_counter.add(1, {"llm.model_chosen": chosen_model})
        inference_latency_histogram.record(inference_duration_ms, {"llm.model_chosen": chosen_model})

        # Add attributes to the gateway span for detailed tracing
        span.set_attribute("llm.model_chosen", chosen_model)
        span.set_attribute("llm.total_inference_time_ms", inference_duration_ms)
        span.set_attribute("llm.total_simulated_cost", total_cost)
        span.set_attribute("gpu.simulated_utilization_at_inference", simulated_gpu_utilization) # Add as span attribute too

        print(f"Gateway processed prompt (len={len(prompt)}): '{prompt[:30]}...' -> Model: {chosen_model}, Cost: ${total_cost:.4f}, Time: {inference_duration_ms:.2f}ms, GPU Util: {simulated_gpu_utilization:.1f}%")
        return response, total_cost

# --- 4. Main Execution ---
if __name__ == "__main__":
    print("Starting LLM Inference Gateway simulation with OpenTelemetry...")
    print("Traces and Metrics will be printed to console.")
    print("-" * 80)

    prompts = [
        "What is the capital of France?",
        "Explain the concept of quantum entanglement in simple terms.",
        "Write a short poem about a cat.",
        "Summarize the economic impact of artificial intelligence on global markets over the next decade, considering both job displacement and new industry creation.",
        "Tell me a joke.",
        "Design a prompt for a generative AI to create a futuristic city skyline with flying cars and bioluminescent flora.",
    ]

    total_simulated_cost_all_inferences = 0.0

    for i, prompt in enumerate(prompts):
        print(f"\n--- Inference Request {i+1} ---")
        response, cost = llm_gateway(prompt)
        total_simulated_cost_all_inferences += cost
        # Simulate some delay between requests to allow metric collection to happen
        time.sleep(random.uniform(0.1, 0.5))

    print("-" * 80)
    print(f"Simulation finished. Total simulated cost for all inferences: ${total_simulated_cost_all_inferences:.4f}")
    
    # Ensure all spans and metrics are exported before exiting
    trace_provider.shutdown()
    metric_provider.shutdown()
    print("OpenTelemetry providers shut down.")
