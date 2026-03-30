@echo off
setlocal

cd /d %~dp0

python benchmark_single_image_inference.py ^
  --image_path "D:\python\paper-reappear\AnomalyCLIP-main\data\mvtec\cable\test\cable_swap\000.png" ^
  --checkpoint_path "D:\python\paper-reappear\AnomalyCLIP-main\checkpoint\noForzen\hsf\epoch_15.pth" ^
  --framework open_vocab ^
  --device cuda:0 ^
  --warmup 10 ^
  --repeat 50 ^
  --output_json "D:\python\paper-reappear\AnomalyCLIP-main\benchmark_results\cable_000_open_vocab.json"

endlocal
