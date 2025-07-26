import os
import torch
import selfies as sf
import numpy as np
from typing import List, Dict, Tuple

class SELFIESProcessor:
    def __init__(self, vocab_file=None, max_vocab_size=1000):
        """
        Optimized SELFIES processor with dynamic vocabulary building.
        
        Args:
            vocab_file: Optional path to pre-built vocabulary file
            max_vocab_size: Maximum vocabulary size to prevent memory issues
        """
        self.max_vocab_size = max_vocab_size
        self.vocab_file = vocab_file
        
        # Special tokens (always first in vocabulary)
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
        else:
            # Start with base vocabulary and build dynamically
            self.charset = self._build_base_charset()
            self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
            self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
            self.vocab_size = len(self.charset)
        
        # Cache for tokenization (significant speedup)
        self._tokenization_cache = {}
        self._encoding_cache = {}
        
    def _build_base_charset(self):
        """Build minimal base character set from most common SELFIES tokens."""
        # Optimized base vocabulary with most common tokens
        base_tokens = [
            # Special tokens first
            '<pad>', '<sos>', '<eos>', '<unk>',
            
            # Most common atoms
            '[C]', '[O]', '[N]', '[F]', '[S]', '[Cl]', '[Br]', '[I]', '[P]',
            '[c]', '[o]', '[n]', '[s]', '[p]',
            
            # Common bonds and modifications
            '[=C]', '[=O]', '[=N]', '[=S]',
            '[#C]', '[#N]',
            
            # Common structural elements
            '[Branch1]', '[Branch2]', '[Ring1]', '[Ring2]',
            '[1]', '[2]', '[3]', '[4]', '[5]', '[6]',
            
            # Common hydrogen patterns
            '[H]', '[CH1]', '[CH2]', '[CH3]', '[NH1]', '[NH2]', '[OH1]',
            
            # Stereochemistry (most common)
            '[C@]', '[C@@]', '[C@H1]', '[C@@H1]',
            
            # Charges (common)
            '[C+]', '[O+]', '[N+]', '[C-]', '[O-]', '[N-]'
        ]
        
        return base_tokens
        
    def _load_vocab(self, vocab_file):
        """Load vocabulary from file."""
        import json
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
        
        self.charset = vocab_data['charset']
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        self.vocab_size = len(self.charset)
        
    def save_vocab(self, vocab_file):
        """Save vocabulary to file for reuse."""
        import json
        vocab_data = {
            'charset': self.charset,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
            'vocab_size': self.vocab_size
        }
        
        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
        with open(vocab_file, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def update_vocab_from_data(self, selfies_strings):
        """
        Update vocabulary based on actual data (more efficient than static vocab).
        
        Args:
            selfies_strings: List of SELFIES strings from the dataset
        """
        import selfies as sf
        from collections import Counter
        
        print("🔄 Building dynamic vocabulary from data...")
        
        # Count all tokens in the dataset
        token_counts = Counter()
        
        for selfies_str in selfies_strings:
            if selfies_str:
                try:
                    tokens = sf.split_selfies(selfies_str)
                    token_counts.update(tokens)
                except:
                    continue
        
        # Keep most frequent tokens up to max_vocab_size
        most_common_tokens = token_counts.most_common(self.max_vocab_size - len(self.special_tokens))
        
        # Build new vocabulary: special tokens + most common tokens
        new_charset = self.special_tokens + [token for token, _ in most_common_tokens]
        
        # Update mappings
        self.charset = new_charset
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.charset)
        
        print(f"✅ Updated vocabulary: {self.vocab_size} tokens")
        print(f"📊 Coverage: {len(most_common_tokens)} unique tokens from data")
        
        # Clear caches after vocab update
        self._tokenization_cache.clear()
        self._encoding_cache.clear()
    
    def selfies_to_tensor(self, selfies_str: str, max_length: int = 128) -> torch.Tensor:
        """
        Convert SELFIES string to tensor representation with caching.
        
        Args:
            selfies_str: SELFIES string to convert
            max_length: Maximum sequence length
            
        Returns:
            torch.Tensor: Token indices tensor
        """
        # Use cache for repeated conversions (significant speedup)
        cache_key = (selfies_str, max_length)
        if cache_key in self._encoding_cache:
            return self._encoding_cache[cache_key].clone()
        
        tokens = list(self.tokenize_selfies(selfies_str))
        
        # Add special tokens
        tokens = ['<sos>'] + tokens + ['<eos>']
        
        # Truncate or pad efficiently
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + ['<eos>']
        else:
            # Pre-allocate list for better performance
            padded_tokens = tokens + ['<pad>'] * (max_length - len(tokens))
            tokens = padded_tokens
        
        # Vectorized conversion to indices (faster than list comprehension)
        indices = [self.char_to_idx.get(token, self.char_to_idx['<unk>']) for token in tokens]
        
        tensor = torch.tensor(indices, dtype=torch.long)
        
        # Cache result if cache isn't too large
        if len(self._encoding_cache) < 10000:
            self._encoding_cache[cache_key] = tensor.clone()
        
        return tensor
    
    def tokenize_selfies(self, selfies_str: str) -> List[str]:
        """
        Optimized SELFIES tokenization with caching.
        
        Args:
            selfies_str: SELFIES string to tokenize
            
        Returns:
            List[str]: List of SELFIES tokens
        """
        # Use cache for repeated tokenizations
        if selfies_str in self._tokenization_cache:
            return self._tokenization_cache[selfies_str]
        
        # Use selfies library for more robust tokenization
        try:
            import selfies as sf
            tokens = sf.split_selfies(selfies_str)
        except:
            # Fallback to manual tokenization
            tokens = self._manual_tokenize(selfies_str)
        
        # Cache result if cache isn't too large
        if len(self._tokenization_cache) < 10000:
            self._tokenization_cache[selfies_str] = tokens
        
        return tokens
    
    def _manual_tokenize(self, selfies_str: str) -> List[str]:
        """Manual tokenization fallback."""
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
        """
        Convert tensor back to SELFIES string efficiently.
        
        Args:
            tensor: Token indices tensor
            
        Returns:
            str: SELFIES string
        """
        # Vectorized conversion (faster than list comprehension)
        if tensor.dim() > 1:
            tensor = tensor.squeeze()
        
        # Convert to list once for efficiency
        indices = tensor.tolist()
        
        # Filter and join in one pass
        tokens = []
        for idx in indices:
            token = self.idx_to_char.get(idx, '<unk>')
            if token not in ['<pad>', '<sos>', '<eos>', '<unk>']:
                tokens.append(token)
            elif token == '<eos>':
                break  # Stop at EOS token
        
        return ''.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def encode_batch(self, selfies_strings: List[str], max_length: int = 128) -> torch.Tensor:
        """
        Encode a batch of SELFIES strings efficiently.
        
        Args:
            selfies_strings: List of SELFIES strings
            max_length: Maximum sequence length
            
        Returns:
            torch.Tensor: Batch of encoded sequences (batch_size, max_length)
        """
        # Pre-allocate tensor for better performance
        batch_size = len(selfies_strings)
        encoded = torch.zeros(batch_size, max_length, dtype=torch.long)
        
        for i, selfies_str in enumerate(selfies_strings):
            encoded[i] = self.selfies_to_tensor(selfies_str, max_length)
        
        return encoded
    
    def decode_batch(self, tensors: torch.Tensor) -> List[str]:
        """
        Decode a batch of tensors to SELFIES strings efficiently.
        
        Args:
            tensors: Batch of token indices (batch_size, seq_len)
            
        Returns:
            List[str]: List of SELFIES strings
        """
        if tensors.dim() == 1:
            tensors = tensors.unsqueeze(0)
        
        decoded = []
        for i in range(tensors.size(0)):
            selfies_str = self.tensor_to_selfies(tensors[i])
            decoded.append(selfies_str)
        
        return decoded
    
    def clear_cache(self):
        """Clear tokenization and encoding caches to free memory."""
        self._tokenization_cache.clear()
        self._encoding_cache.clear()
    
    def get_cache_stats(self):
        """Get cache statistics for monitoring."""
        return {
            'tokenization_cache_size': len(self._tokenization_cache),
            'encoding_cache_size': len(self._encoding_cache)
        }

def create_condition_vector(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    """Create a placeholder condition vector for now."""
    # TODO: Implement actual conditioning based on molecular properties
    return torch.randn(batch_size, latent_dim).to(device)