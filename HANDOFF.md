# P06 handoff

The official DDIM solver and LCM boundary functions have been integrated at a fixed revision. Both schedule/metric tests pass on Python 3.11. Shared SD1.5 and the public LCM-LoRA are being downloaded with checksum verification before the GPU run.

Next command: `python run.py demo --base-path <sd15> --adapter-path <lcm15>`, then `python run.py reproduce --base-path <sd15> --adapter-path <lcm15>`. Record the actual adapter update, full inference grid and measured timing before release.
