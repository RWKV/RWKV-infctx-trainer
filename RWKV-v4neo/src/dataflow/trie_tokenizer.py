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
    def __init__(self, file_name):
        self.vocab_size = 65525
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

from multiprocessing.pool import ThreadPool
from threading import Thread
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

num_cpus = cpu_count()
shared_executor = ProcessPoolExecutor(max_workers=num_cpus)

class MT_TRIE_TOKENIZER():
    def __init__(self, filename):
        # self.trie_tokenizer = None
        self.trie_tokenizer = TRIE_TOKENIZER(filename)
    
    def encode(self, src):
        return self.trie_tokenizer.encode(src)
        # return shared_executor.submit(self.trie_tokenizer.encode, src).result()
        # return [0]
        # twrv = ThreadWithReturnValue(target=self.trie_tokenizer.encode, args=(src,))
        # twrv.start()
        # return twrv.join()