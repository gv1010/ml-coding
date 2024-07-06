import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(["[SOS]"])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(["[EOS]"])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(["[PAD]"])], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang_lang]
        
        enc_input_tokens = self.tokenizer_src.encoder(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encoder(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is tooo long")
        
        # adding eos and sos and pad

        encoder_input = torch.cat([
                        self.sos_token,
                        torch.tensor(enc_input_tokens),
                        self.eos_token,
                        torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
                        ])
        
        decoder_input = torch.cat([
                        self.sos_token,
                        torch.tensor(enc_input_tokens),
                        self.eos_token,
                        torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
                        ])