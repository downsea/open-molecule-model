import torch
import selfies as sf
import numpy as np
from typing import List, Dict, Tuple

class SELFIESProcessor:
    def __init__(self):
        self.charset = self._build_charset()
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.charset)
        
    def _build_charset(self):
        """Build character set from common SELFIES tokens"""
        # Common SELFIES tokens
        charset = [
            '[C]', '[O]', '[N]', '[F]', '[S]', '[Cl]', '[Br]', '[I]', '[P]', '[B]',
            '[c]', '[o]', '[n]', '[s]', '[p]',
            '[=C]', '[=O]', '[=N]', '[=S]', '[=P]',
            '[#C]', '[#N]',
            '[C@H1]', '[C@@H1]', '[Branch1]', '[Branch2]', '[Branch3]',
            '[Ring1]', '[Ring2]', '[Ring3]', '[=Branch1]', '[=Branch2]',
            '[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]',
            '[H]', '[\\H]', '[/H]', '[\\\\H]', '[//H]',
            '[C@]', '[C@@]', '[NH1]', '[NH2]', '[NH3]',
            '[CH1]', '[CH2]', '[CH3]', '[OH1]', '[SH1]',
            '[PH1]', '[PH2]', '[BH1]', '[BH2]', '[BH3]',
            '[Si]', '[Se]', '[Ge]', '[As]', '[Se]', '[Te]',
            '[=c]', '[=n]', '[=o]', '[=s]',
            '[#c]', '[#n]', '[#o]', '[#s]',
            '[C-]', '[O-]', '[N-]', '[S-]', '[P-]',
            '[C+]', '[O+]', '[N+]', '[S+]', '[P+]',
            '[/C]', '[\\C]', '[/N]', '[\\N]', '[/O]', '[\\O]',
            '[/c]', '[\\c]', '[/n]', '[\\n]', '[/o]', '[\\o]',
            '[\\C@]', '[//C@]', '[\\C@@]', '[//C@@]',
            '[\\C]', '[//C]', '[\\N]', '[//N]', '[\\O]', '[//O]',
            '[\\S]', '[//S]', '[\\P]', '[//P]',
            '[\\c]', '[//c]', '[\\n]', '[//n]', '[\\o]', '[//o]',
            '[\\s]', '[//s]', '[\\p]', '[//p]',
            '[C@@H1]', '[C@H1]', '[C@@H2]', '[C@H2]', '[C@@H3]', '[C@H3]',
            '[NH1+]', '[NH2+]', '[NH3+]', '[OH1+]', '[SH1+]',
            '[CH1+]', '[CH2+]', '[CH3+]', '[PH1+]', '[PH2+]', '[PH3+]',
            '[SiH1]', '[SiH2]', '[SiH3]', '[SiH4]',
            '[SeH1]', '[SeH2]', '[GeH1]', '[GeH2]', '[GeH3]', '[GeH4]',
            '[AsH1]', '[AsH2]', '[AsH3]', '[AsH4]',
            '[\\Si]', '[/Si]', '[\\\\Si]', '[//Si]',
            '[\\Se]', '[/Se]', '[\\\\Se]', '[//Se]',
            '[\\Ge]', '[/Ge]', '[\\\\Ge]', '[//Ge]',
            '[\\As]', '[/As]', '[\\\\As]', '[//As]',
            '[\\Te]', '[/Te]', '[\\\\Te]', '[//Te]',
            '[S@]', '[S@@]', '[S@H1]', '[S@@H1]',
            '[P@]', '[P@@]', '[P@H1]', '[P@@H1]',
            '[N@]', '[N@@]', '[N@H1]', '[N@@H1]',
            '[O@]', '[O@@]',
            '[Branch1]', '[Branch2]', '[Branch3]',
            '[=Branch1]', '[=Branch2]', '[=Branch3]',
            '[#Branch1]', '[#Branch2]', '[#Branch3]',
            '[Ring1]', '[Ring2]', '[Ring3]', '[=Ring1]', '[=Ring2]', '[=Ring3]',
            '[#Ring1]', '[#Ring2]', '[#Ring3]',
            '[.]', '[_]', '[\\]', '[/]', '[\\\\]', '[//]',
            '[1]', '[2]', '[3]', '[4]', '[5]', '[6]', '[7]', '[8]', '[9]',
            '[10]', '[11]', '[12]', '[13]', '[14]', '[15]',
            '[16]', '[17]', '[18]', '[19]', '[20]',
            '[21]', '[22]', '[23]', '[24]', '[25]',
            '[26]', '[27]', '[28]', '[29]', '[30]',
            '[31]', '[32]', '[33]', '[34]', '[35]',
            '[36]', '[37]', '[38]', '[39]', '[40]',
            '[41]', '[42]', '[43]', '[44]', '[45]',
            '[46]', '[47]', '[48]', '[49]', '[50]',
            '[51]', '[52]', '[53]', '[54]', '[55]',
            '[56]', '[57]', '[58]', '[59]', '[60]',
            '[61]', '[62]', '[63]', '[64]', '[65]',
            '[66]', '[67]', '[68]', '[69]', '[70]',
            '[71]', '[72]', '[73]', '[74]', '[75]',
            '[76]', '[77]', '[78]', '[79]', '[80]',
            '[81]', '[82]', '[83]', '[84]', '[85]',
            '[86]', '[87]', '[88]', '[89]', '[90]',
            '[91]', '[92]', '[93]', '[94]', '[95]',
            '[96]', '[97]', '[98]', '[99]', '[100]',
            '[101]', '[102]', '[103]', '[104]', '[105]',
            '[106]', '[107]', '[108]', '[109]', '[110]',
            '[111]', '[112]', '[113]', '[114]', '[115]',
            '[116]', '[117]', '[118]', '[119]', '[120]',
            '[121]', '[122]', '[123]', '[124]', '[125]',
            '[126]', '[127]', '[128]', '[129]', '[130]',
            '[131]', '[132]', '[133]', '[134]', '[135]',
            '[136]', '[137]', '[138]', '[139]', '[140]',
            '[141]', '[142]', '[143]', '[144]', '[145]',
            '[146]', '[147]', '[148]', '[149]', '[150]',
            '<pad>', '<sos>', '<eos>', '<unk>'
        ]
        
        return charset
    
    def selfies_to_tensor(self, selfies_str: str, max_length: int = 128) -> torch.Tensor:
        """Convert SELFIES string to tensor representation."""
        tokens = self.tokenize_selfies(selfies_str)
        
        # Add special tokens
        tokens = ['<sos>'] + tokens + ['<eos>']
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + ['<eos>']
        else:
            tokens = tokens + ['<pad>'] * (max_length - len(tokens))
        
        # Convert to indices
        indices = [self.char_to_idx.get(token, self.char_to_idx['<unk>']) for token in tokens]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def tokenize_selfies(self, selfies_str: str) -> List[str]:
        """Tokenize SELFIES string into individual tokens."""
        tokens = []
        i = 0
        while i < len(selfies_str):
            if selfies_str[i] == '[':
                # Find closing bracket
                j = i + 1
                while j < len(selfies_str) and selfies_str[j] != ']':
                    j += 1
                if j < len(selfies_str):
                    tokens.append(selfies_str[i:j+1])
                    i = j + 1
                else:
                    tokens.append(selfies_str[i:])
                    break
            else:
                # Single character token
                tokens.append(selfies_str[i])
                i += 1
        return tokens
    
    def tensor_to_selfies(self, tensor: torch.Tensor) -> str:
        """Convert tensor back to SELFIES string."""
        tokens = [self.idx_to_char[idx.item()] for idx in tensor]
        
        # Remove special tokens and join
        filtered_tokens = []
        for token in tokens:
            if token in ['<pad>', '<sos>', '<eos>', '<unk>']:
                continue
            filtered_tokens.append(token)
        
        return ''.join(filtered_tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def encode_batch(self, selfies_strings: List[str], max_length: int = 128) -> torch.Tensor:
        """Encode a batch of SELFIES strings."""
        encoded = []
        for selfies_str in selfies_strings:
            tensor = self.selfies_to_tensor(selfies_str, max_length)
            encoded.append(tensor)
        return torch.stack(encoded)
    
    def decode_batch(self, tensors: torch.Tensor) -> List[str]:
        """Decode a batch of tensors to SELFIES strings."""
        decoded = []
        for tensor in tensors:
            selfies_str = self.tensor_to_selfies(tensor)
            decoded.append(selfies_str)
        return decoded

def create_condition_vector(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    """Create a placeholder condition vector for now."""
    # TODO: Implement actual conditioning based on molecular properties
    return torch.randn(batch_size, latent_dim).to(device)