# !NOTE

The notebooks, were copied over from the original `rwkv5x-tokenshift-branch` branch, as it is

As such most of the paths have not been updated, and may not work without some modification first.

---

The following are the main list of TokenShift experiments

**TokenShift-A**

Initial POC
- 12 wavenet layers
- 2560 embedding size
- 1.3B params

**TokenShift-B**

Refinement of the POC model, with additional non-wavenet layers.
- 12 wavenet layers
- 12 normal layers
- 1024 embedding size
- 430M params

**TokenShift-C**

Scaling up of TokenShift-B, with larger dimension size, an identified bottleneck in the upper layer channels.
- 12 wavenet layers
- 12 normal layers
- 2048 embedding size
- 1.5B params

This has the same layout as the existing RWKV 1B5 models

**TokenShift-D**

Meant to be a replication of the L96-D1024 experiment, with the new 12 layer wavenet. This is used to measure the, difference in performance of layer vs dimensions with the new wavenet structure

- 12 wavenet layers
- 84 normal layers
- 1024 embedding size
- 1.4B params

**TokenShift-E**

A reduced version of RWKV 14B L40 D5120 model. This allow us to have a stron approximate to the memory capacity performance of the raven model.

- 12 wavenet layers
- 12 normal layers
- 5120 embedding size
- 3B params

**TokenShift-K**

Replication of tokenshift-B with reduced dataset size. The goal was initially to streamline the training process of tokenshift models.
It instead open more questions in the process.

This may need additional experimentation varients.
