import os
from typing import TextIO
import pickle 
import regex as re 

class Tokenizer():
    def __init__(self,
                 vocab:dict[int,bytes],
                 merges:list[tuple[bytes,bytes]],
                 special_tokens:list[str]|None) -> None:
        self.vocab = vocab 
        self.vocab_to_index = {v:k for k,v in vocab.items()}
        self.merges = merges
        if isinstance(special_tokens,list):
            if "<|endoftext|>" not in special_tokens:
                special_tokens.append("<|endoftext|>")
            special_tokens=sorted(special_tokens,key = len,reverse=True) 
            for special_token in special_tokens:
                if special_token.encode('utf-8') not in self.vocab_to_index:
                    self.vocab_to_index[special_token.encode('utf-8')]=len(self.vocab)
                    self.vocab[len(self.vocab)] = special_token.encode('utf-8')
        
        self.special_tokens = special_tokens 
    
    @classmethod
    def load_from_pickle(cls,file_path):
        try:
            with open(file_path,'rb') as f:
                data = pickle.load(f)
            return data
        except FileNotFoundError:
            print(f"file {file_path} not found!")
            return None 
        except Exception as e:
            print(f"Fail to read:{e}")
            return None 

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = Tokenizer.load_from_pickle(vocab_filepath)
        merges = Tokenizer.load_from_pickle(merges_filepath)
        assert vocab !=None and merges != None 

        return cls(vocab,merges,special_tokens)

    def pretokenize(self,text:str):
        pre_tokens = {}
        pre_token_seq = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        special_tokens_seq = []

        if self.special_tokens:
            concat_special_tokens = '|'.join([re.escape(special_token) for special_token in self.special_tokens])
            split_chunk = re.split(concat_special_tokens,text)
            matches = re.finditer(concat_special_tokens,text)
            special_tokens_seq = [match.group() for match in matches]
        else:
            split_chunk = [text]
        
        i=0

        for seg in split_chunk:
            pre_token_iter = re.finditer(PAT,seg)
            for pre_token in pre_token_iter:
                pre_token = pre_token.group() # return the whole matched string 
                pre_token_result = [bytes([byte]) for byte in pre_token.encode('utf-8')]
                if pre_token not in pre_tokens:
                    pre_tokens[pre_token] = pre_token_result
                pre_token_seq.append(pre_token)

            if i<len(special_tokens_seq):
                pre_token_seq.append(special_tokens_seq[i])
            i+=1
        return pre_tokens,pre_token_seq,special_tokens_seq
    
    def merge(self,pre_tokens):
        for merge in self.merges:
            for pre_token,pre_token_result in pre_tokens.items():
                i = 0
                while i<len(pre_token_result)-1:
                    bp = (pre_token_result[i],pre_token_result[i+1])
                    if bp == merge:
                        pre_token_result[i] = bp[0]+bp[1]
                        pre_token_result.pop(i+1)
                    i+=1
                pre_tokens[pre_token] = pre_token_result 
        
        return pre_tokens



    def encode(self, text: str) -> list[int]:
        pre_tokens,pre_token_seq,special_tokens_list = self.pretokenize(text)
        pre_tokens = self.merge(pre_tokens)
        pre_tokens_to_ids = {}
        token_ids = []

        for pre_token,pre_token_result in pre_tokens.items():
            pre_token_ids = []
            for byte in pre_token_result:
                pre_token_ids.append(self.vocab_to_index[byte])
            
            pre_tokens_to_ids[pre_token] = pre_token_ids
        for special_token in special_tokens_list:
            if special_token not in pre_tokens_to_ids:
                pre_tokens_to_ids[special_token] = [self.vocab_to_index[special_token.encode('utf-8')]]
        
        for pre_token in pre_token_seq:
            token_ids.extend(pre_tokens_to_ids[pre_token])
        return token_ids




    def encode_iterable(self, iterable: TextIO):
        chunk_size = 4096
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        file_pos = 0
        

        while True:
            chunk= iterable.read(chunk_size)
            pos = iterable.tell()

            if not chunk:
                break

            is_last_chunk = len(chunk) < chunk_size
            original_chunk = chunk 

            if not is_last_chunk:
                if self.special_tokens:
                    concat_special_tokens = '|'.join([re.escape(special_token) for special_token in self.special_tokens]) 
                    end_chunk = chunk[-2*len(max(self.special_tokens,key=len)):] 
                    matches = list(re.finditer(concat_special_tokens, end_chunk))
                    if len(matches) > 0:
                        chunk = chunk[:matches[0].span()[0] ]
                    else:
                        chunk = chunk[:-len(max(self.special_tokens,key=len))]
                        pretokens = re.findall(PAT,chunk)
                        if pretokens:
                            endtoken_len = len(pretokens[-1])
                            chunk = chunk[:-endtoken_len]
                else:
                    pretokens = re.findall(PAT,chunk)
                    if pretokens:
                        endtoken_len = len(pretokens[-1])
                        chunk = chunk[:-endtoken_len]
                
                if not chunk:
                    chunk = original_chunk
                
            chunk_len = len(chunk)

            if chunk_len == 0:
                break

            iterable.seek(file_pos)
            iterable.read(chunk_len)
            file_pos = iterable.tell()
            iterable.seek(file_pos)
            token_ids = self.encode(chunk)
            for token_id in token_ids:
                yield token_id


    def decode(self,ids:list[int]) ->str:
        decode_bytes = b""
        for id in ids:
            decode_bytes += self.vocab[id]
        decode_str = decode_bytes.decode('utf-8',errors='replace') 
        return decode_str 
 