# RWKV GPU scheduling

RWKV is constantly being developed and iterated on. Beyond the large GPU cluster training (aka: a100s) which is provided for limited burst of time by eleutherAI or other sponsors for the foundation models.

We have much smaller nodes, used for either experimenting on RWKV model changes, for the next generation model.

In between these iteration, of setting up new training runs, interprating of data, and sometimes just bugs/errors that happens when the writer is asleep (more often then you think). The GPU ends up being idle for short stretches of 2-6 hours while those issues are being sorted out (or until the writer wakes up).

As such we are opening up training notebook submissions (via PR), for queing up lower priority runs within the cluster. To ensure the GPUs is "not wasted".

This is made avaliable in 2 major varients

- [RWKV Community Training](./RWKV-community-training.md): Want to finetune an existing model? For a good use case for the community
- [RWKV-X playground](./RWKV-x-playground-training.md): Have an experimental idea that can help improve the model? Provide the code change, and we will try to run a 0.1 - 1.5B model for it. To test your hypothesis.

For most parts community model will be priotised based on
- Positive impact to the community
- How short it is expected to be for each part of the run
- Core RWKV-X experiments will typically take priority

# Who manages the GPU scheduling?

All community based GPU scheduling will be handled by @picocreator
In the future as we scale up, additional members of the community will be designated access and permission to cordinate such runs.

# Hey I got spare GPU compute, can I donate to the pool!

YES! In general if you have a modern GPU (ie. 30XX or 40XX, or better). Your nodes can be useful for various use cases

- Single 16-20GB GPUs is serverly needed to help get our CI/CD pipeline running
- Pairs of 24GB GPUs or higher, is very suitible for finetuning 1.5B models experiments
- Got spare 4-8 x GPUs or >24GB vram GPUS? These are in dire need to help speed up overall experimentation on potentially larger models

As all scheduling of GPU runs will be done via github actions, to contribute your GPU all you need is to contact @picocreator via discord with your specs. And he will provide you the docker command to run on your system. 

Which will end up looking something like this like this

```bash
docker run --name rwkv-github-runner \
    --gpus all \
    --storage-opt size=250G \
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