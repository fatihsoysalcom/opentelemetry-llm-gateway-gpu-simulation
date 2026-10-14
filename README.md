# OpenTelemetry LLM Gateway GPU Simulation

This example demonstrates an LLM (Large Language Model) inference gateway instrumented with OpenTelemetry. It simulates routing requests to different LLM backends based on prompt complexity and captures performance metrics, simulated costs, and 'OTel-native' simulated GPU utilization. The output shows how OpenTelemetry traces and metrics can provide transparency into AI inference costs and resource usage.

## Language

`python`

## How to Run

1. Save the code as `main.py`.
2. Install OpenTelemetry SDK: `pip install opentelemetry-api opentelemetry-sdk`
3. Run the script: `python main.py`

## Original Article

This example accompanies the Turkish article: [Yüksek Yapay Zeka Çıkarım Maliyetlerine Son: OTel-Native GPU Profiler ve LLM Ağ Geçidi](https://fatihsoysal.com/blog/yuksek-yapay-zeka-cikarim-maliyetlerine-son-otel-native-gpu-profiler-ve-llm-ag-gecidi/).

## License

MIT — see [LICENSE](LICENSE).
