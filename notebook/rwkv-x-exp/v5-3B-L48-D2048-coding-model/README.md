# L48-D2048 coding model

The general design of this model, is the 1.5B model upsized to 3B, with extended layers to allow the model to handle context sizes of 4k to 8k with ease. Training is done with the world tokenizer

It generally has the following phases

- enwiki_100k
- memory tuning (2 stages)
- slimpajama
- starcoder

Once completed it can be instruction tuned for further use cases (ie. chat)