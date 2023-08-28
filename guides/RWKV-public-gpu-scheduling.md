# RWKV GPU scheduling

RWKV is constantly being developed and iterated on. Beyond the large GPU cluster training (aka: a100s) which is provided for limited burst of time by eleutherAI or other sponsors for the foundation models.

We have much smaller nodes, used for either experimenting on RWKV model changes, for the next generation model.

To help the community and model design moving forward we are opening access for 
- [RWKV Community Training](./RWKV-community-training.md): training of community models, priotised by positive impact to the community (ie. minority language models), or creative use ideas (ie. music gen)
- [RWKV-X playground](./RWKV-x-playground-training.md): R&D into the next generation of RWKV ideas. Where code and benchmarks will be the proving ground. To test your hypothesis.

Most of these runs will be executed between higher priority runs by existing researcher / maintainers.

The following is an example of such a run, scheduled via our github actions
- [Github actions link](https://github.com/RWKV/RWKV-infctx-trainer/actions/runs/5983559635/job/16233828093)
- [Notebook used](https://huggingface.co/rwkv-x-dev/rwkv-x-playground/blob/main/experiment/rwkv-x-exp/v5-headsize32/v5-L6-D2048-E1e-1-ctx4k-part1.ipynb)
- [Uploaded models](https://huggingface.co/rwkv-x-dev/rwkv-x-playground/tree/main/experiment/rwkv-x-exp/v5-headsize32)

For most parts these runs will be priotised based on
- Positive impact to the community
- How short it is expected to be for each part of the run

Our GPUs in the community pool sponsors can be found here: [GPU sponsor list](./GPU-sponsor-list.md)

# Who manages the GPU scheduling?

All community based GPU scheduling will be handled by @picocreator
In the future as we scale up, additional members of the community will be designated access and permission to cordinate such runs.

# Hey I got spare GPU compute, can I donate to the pool!

YES! In general if you have a modern GPU (ie. 30XX or 40XX, or better). Your nodes can be useful for various use cases

- Single 16-20GB GPUs is serverly needed to help get our CI/CD pipeline running
- Pairs of 24GB GPUs or higher, is very suitible for finetuning 1.5B models experiments
- Got spare 4-8 x GPUs or >24GB vram GPUS? These are in dire need to help speed up overall experimentation on potentially larger models

> Do allocate atleast 500GB of storage space, as the datasets & checkpoints can get extreamly large especially when stacked up
> even if we are doing a hard reset between every run. If you have <500GB of space, do let us know and we will adjust the runs accordingly.

As all scheduling of GPU runs will be done via github actions, to contribute your GPU all you need is to contact @picocreator via discord with your specs. And he will provide you the docker command to run on your system. 

Which will end up looking something like this like this

```bash
docker run --name rwkv-github-runner \
    --gpus all \
    --storage-opt size=500G \
    --env RUNNER_NAME="picocreator-1x3090" \
    --env RUNNER_LABELS="1x3090,1x24vgb,100GB" \
    --env RUNNER_TOKEN="<SECRET KEY>" \
    -t ghcr.io/picocreator/rwkv-lm-lora:github-worker-cuda-11-8
```

Additionally, your GPU contribution do not need to be exclusive 24/7. As we will resecheduled any interrupted runs.

However do try to provide it in burst of 12 hours at a time approximately, as this will help ensure atleast one or more individual runs will compelete properly, during the allocation period.

To ensure no abuse / malicious runs, we will be vetting all notebooks accordingly prior to running.

You may also indicate to us, if you want to limit your GPU contribution to core members experimental runs (and not community runs), and we will setup the runner and respect your decision accordingly.

Sponsors will be tracked in the [GPU sponsor list](./GPU-sponsor-list.md), unless they wish to be anonymous
