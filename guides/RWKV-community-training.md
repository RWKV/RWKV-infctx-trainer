# RWKV community model training

Want to train a new model, or finetune ?

In general, models with a positive impact to RWKV community (eg. specific language models), or new unexplored use cases, will be priotised over other models, and scheduled between RWKV-x researchers runs.

# Steps for the process

1) Clone the `rwkv-community-training` branch : https://github.com/RWKV/RWKV-infctx-trainer/
2) Inside the `notebook/community` directory, create a folder for your training process, and prepare the notebook & config. See an example here : https://github.com/RWKV/RWKV-infctx-trainer/tree/main/notebook/community/example
    - your must output your finished model into the "model" folder, this is what gets scanned for upload
    - your dataset should be downloadable via huggingface / a link, do not commit your dataset into the repo
    - any required models should also be donwloaded as part of the notebook
    - use the example settings for datapath & checkpoints
    - keep the notebook name as `training-notebook`
    - for larger runs, split it into chunks with `-part1` to `-partx` suffixes
3) Test and iterate your changes with a smaller model locally if possible
    - Important to note: we may not have time to debug issues with your notebook, if it does not run, we will skip and move on to the next request in queue
4) Perform a pull request to `rwkv-community-training` with the title `[training request] description`, you must include your RWKV discord handle, and a description of what you are trying to train, so we can evaulate and schedule priority accordingly
    - Note that all followup will be done via discord.
    - We will very likely rename, and reorganize your files, and if we do so, we will keep you updated. 
    - Once completed, all training runs results and output will be uploaded to hugging face. Example: https://huggingface.co/rwkv-x-dev/rwkv-x-playground/tree/main/experiment/rwkv-x-exp/v5-headsize32
    - for maintainers / users with direct write access, you can probably just skip the PR, and just do the runs from your branch if desired.

> Note: If you intend to use blinks official trainer, in your notebook. Do clone the official trainer, and make sure the output model is saved into the "model" directory.
> As the github action will **only** be saving the `*.pth` files inside the "model" directory of this repo.

# Restrictions on the training process

- Generally, keep your model size <= 1.5B - as we only have up to 3090s currently (not a hard-rule, but anything bigger typically take days)
    - For 3B/7B models, because there is a huge performance penalty for deepspeed 3, try to keep your dataset conservative in size, to keep the run under 12 hours.
- Keep your training runs under 12 hours, we may priotize shorter runs over longer runs (case by case basis)
- We have the right of refusal, and have no obligations. Existing runs may take priority
- Everything will be done in public. NSFW / trolling runs will not be accepted
