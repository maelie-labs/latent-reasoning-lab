# Compute-to-Threshold & Resource Trade-Off Table

| Experimental Arm | Pass@1 Acc (%) | Mean Think Tokens / Latents | Reasoning TFLOPs | Peak Reasoning KV (KiB) | Mean Latency (ms) | Latency / Correct Query (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm4_Unconstrained** | 86.70% | 3088.0 | 10.499 TFLOPs | 362,653 KiB | 83,193.2 ms | 95,955.2 ms |
| **Arm2_K32** | 74.70% | 32.0 | 0.109 TFLOPs | 20,384 KiB | 11,982.1 ms | 16,040.3 ms |
| **Arm2_K6** | 73.00% | 6.0 | 20.4 GFLOPs | 17,472 KiB | 11,660.3 ms | 15,973.1 ms |