# Few-step diffusion distillation handoff

Formal run completed on Windows using SD1.5 FP16, an RTX 4060 Laptop GPU, the public LCM adapter and a rank-two adapter trained for 32 updates. Adapter L2 update was 0.370499. Each evaluation arm loaded a fresh copy of the base to isolate its adapter. Every trained parameter tensor was checked after serialization and reload.

Latest verified commands: `python run.py reproduce --base-path <sd15> --adapter-path <lcm15> --device cuda`, then `python run.py evaluate`. Artifacts: results/measurements.json, 54 prediction images, results/summary.csv, results/comparison.png and results/trained.safetensors. All eligible photographs, prompt splits, seed and training budget are in configs/protocol.json.

Use `python run.py demo` for the 512 px three-way comparison. The clean GitHub clone at 50ac917d passed the demo, unit tests, full retraining and both saved-prediction evaluations. Fixed-model images matched 36/36; the maximum independently retrained MSE delta was 0.001070. Both runs are archived. Next local command: python run.py demo. Timing depends on hardware and other workloads.
