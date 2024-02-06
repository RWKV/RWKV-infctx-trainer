# To run the github runner

Use the following format

```
docker run -it --gpus all \
  --env RUNNER_LABELS='your-labels-here' \
  --env RUNNER_NAME='your-runner-name-here' \
  --env RUNNER_TOKEN='your-runner-token-here' \
  --env RUNNER_REPO_URL='https://github.com/RWKV' \
  --name RWKV-GH-GPU-Runner \
  ghcr.io/rwkv/rwkv-infctx-trainer:github-worker-cuda-12-1
```

For labels use the following
- {manufacturer}-gpu
- gpu-vram-{16/24/32/40/48/80}
- gpu-vram-eq-{exact-vram-class}
- gpu-count-{1/2/4/8}
- gpu-{model-label}
- gpu-{1/2/4/8}x{model-label}
- any-gpu
- gpu-count-any

For `gpu-vram-x`, include lower bound numbers that is less or equal to your GPU vram.
This allow us to make use of GPUs with higher vram, for tasks with lower requirements in the Q.

Ensure you have 1.5 x RAM to total vram ratio, and atleast 2 vCPU per GPU

So for example, to run on an 8 x 4090 server, you can use the following
> The label does get long

```
docker run -it --gpus all \
  -e RUNNER_LABELS='nvidia-gpu,gpu-vram-16,gpu-vram-24,gpu-vram-eq-24,gpu-count-8,gpu-4090,gpu-8x4090,any-gpu,gpu-count-any' \
  -e RUNNER_NAME='your-runner-name-here' \
  -e RUNNER_TOKEN='your-runner-token-here' \
  -e RUNNER_REPO_URL='https://github.com/RWKV' \
  --name RWKV-GH-GPU-Runner \
  ghcr.io/rwkv/rwkv-infctx-trainer:github-worker-cuda-12-1
```

Consult the RWKV group runner manager : @picocreator
If you like to donate your GPU compute to the pool
The runner name, is meant to make it easy to identify the GPU source, and isolate issues if needed.
