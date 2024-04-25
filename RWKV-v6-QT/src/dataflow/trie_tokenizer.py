########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#
# Original source code for TRIE_TOKENIZER: 
# https://github.com/BlinkDL/ChatRWKV/blob/1e408fe50d7059bfa4319835f22cbfa88d8ad14e/rwkv_pip_package/src/rwkv/rwkv_tokenizer.py
# https://github.com/TkskKurumi/ChatRWKV-TRIE-Tokenizer
########################################################################################################

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

class TRIE_TOKENIZER():
    def __init__(self, file_name, add_endoftext_token=True):
        self.vocab_size = 65536
        self.idx2token = {}
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x
        
        # Add the <|endoftext|> token overwrite
        if add_endoftext_token:
            self.idx2token[0] = b'<|endoftext|>'
        
        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))            
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.idx2token

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()

########################################################################################################
# The following is a MT extension of the trie tokenizer
########################################################################################################

# Get the current file dir
import os
DATAFLOW_DIR = os.path.dirname(os.path.abspath(__file__))

# We use this in global, easier for HF to handle for arrow functions ??
WORLD_TOKENIZER_WITH_EOT = None
WORLD_TOKENIZER_NO_EOT = None

# Setup the local world tokenizer if needed and return it
def get_world_tokenizer(world_add_endoftext_token=True):
    global WORLD_TOKENIZER_WITH_EOT, WORLD_TOKENIZER_NO_EOT
    if world_add_endoftext_token:
        if WORLD_TOKENIZER_WITH_EOT is None:
            WORLD_TOKENIZER_WITH_EOT = TRIE_TOKENIZER(os.path.join(DATAFLOW_DIR, "./rwkv_vocab_v20230424.txt"), add_endoftext_token=True)
        return WORLD_TOKENIZER_WITH_EOT
    else:
        if WORLD_TOKENIZER_NO_EOT is None:
            WORLD_TOKENIZER_NO_EOT = TRIE_TOKENIZER(os.path.join(DATAFLOW_DIR, "./rwkv_vocab_v20230424.txt"), add_endoftext_token=False)
        return WORLD_TOKENIZER_NO_EOT

# Provide a global function for the world tokenizer
def world_tokenizer_encode(src, world_add_endoftext_token=True):
    return get_world_tokenizer(world_add_endoftext_token=world_add_endoftext_token).encode(src)

########################################################################################################
# Tensor specific tokenizer
########################################################################################################

import torch
import numpy as np

class MT_TRIE_TOKENIZER():
    def __init__(self, filename):
        # self.trie_tokenizer = None
        self.trie_tokenizer = TRIE_TOKENIZER(filename)
    
    def encode(self, src):
        # Get the encoded tokens, however because this 
        # is a mix of int and longs, we need to convert 
        # to the torch formatting
        raw_tokens = self.trie_tokenizer.encode(src)
        tokens_len = len(raw_tokens)

        # lets setup the tensors
        tokens = torch.zeros(tokens_len, dtype=torch.long)

        # now we need to convert the raw tokens to the torch format
        for i in range(tokens_len):
            tokens[i] = raw_tokens[i]

        # Return tokens
        return tokens

    def decode(self, tokens):
        # We ensure that the tokens are passed as list of int/long 
        # and not as a torch tensor/numpy array
        tokens_len = len(tokens)

        # The clean token array to build
        clean_tokens = []

        # Now we need to convert the tokens to the raw tokens
        for i in range(tokens_len):
            # If torch
            if isinstance(tokens[i], torch.Tensor):
                clean_tokens.append(tokens[i].item())
            # If numpy
            elif isinstance(tokens[i], np.ndarray):
                clean_tokens.append(tokens[i].item())
            # If int/long
            elif isinstance(tokens[i], int) or isinstance(tokens[i], long):
                clean_tokens.append(tokens[i])
            # If unknown
            else:
                raise Exception(f"Unknown token type: {type(tokens[i])}")

        # Decode and return
        return self.trie_tokenizer.decode(clean_tokens)
        
