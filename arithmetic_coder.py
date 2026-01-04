"""
Arithmetic Coding Implementation for CommaVQ Compression

This implements range-based arithmetic coding with 32-bit integer precision.
It's designed to work with probability distributions from neural networks.
"""

import numpy as np
from typing import List, Tuple


class ArithmeticCoder:
    """
    Arithmetic encoder/decoder using integer arithmetic.
    
    Uses 32-bit precision with renormalization to handle arbitrary-length sequences.
    """
    
    # Precision constants (32-bit)
    PRECISION = 32
    WHOLE = (1 << 32) - 1      # 0xFFFFFFFF
    HALF = 1 << 31             # 0x80000000  
    QUARTER = 1 << 30          # 0x40000000
    
    def __init__(self, precision_bits: int = 16):
        """
        Args:
            precision_bits: Bits used for probability precision (default 16)
                           Higher = more accurate but slower
        """
        self.precision_bits = precision_bits
        self.prob_scale = 1 << precision_bits  # Scale probabilities to integers
        
    def probs_to_cdf(self, probs: np.ndarray) -> np.ndarray:
        """
        Convert probability distribution to cumulative distribution function.
        
        Args:
            probs: Array of probabilities, shape (vocab_size,), sums to 1
            
        Returns:
            cdf: Array of shape (vocab_size + 1,) where:
                 cdf[0] = 0
                 cdf[i] = sum(probs[0:i]) scaled to integers
                 cdf[-1] = prob_scale
        """
        # Scale probabilities to integers
        scaled = (probs * self.prob_scale).astype(np.int64)
        
        # Ensure no zeros (arithmetic coding needs non-zero probabilities)
        scaled = np.maximum(scaled, 1)
        
        # Renormalize to exactly prob_scale
        total = scaled.sum()
        if total != self.prob_scale:
            # Adjust the largest probability to make sum exact
            diff = self.prob_scale - total
            max_idx = np.argmax(scaled)
            scaled[max_idx] += diff
            
        # Compute CDF
        cdf = np.zeros(len(probs) + 1, dtype=np.int64)
        cdf[1:] = np.cumsum(scaled)
        
        return cdf


class ArithmeticEncoder(ArithmeticCoder):
    """Encodes a sequence of symbols given their probability distributions."""
    
    def __init__(self, precision_bits: int = 16):
        super().__init__(precision_bits)
        self.reset()
        
    def reset(self):
        """Reset encoder state for a new sequence."""
        self.low = 0
        self.high = self.WHOLE
        self.pending_bits = 0
        self.output_bits: List[int] = []
        
    def encode_symbol(self, symbol: int, cdf: np.ndarray):
        """
        Encode a single symbol.
        
        Args:
            symbol: The symbol to encode (0 to vocab_size-1)
            cdf: Cumulative distribution function from probs_to_cdf()
        """
        # Current range
        range_size = self.high - self.low + 1
        
        # Narrow the range based on symbol's probability interval
        # symbol occupies [cdf[symbol], cdf[symbol+1]) out of [0, cdf[-1])
        self.high = self.low + (range_size * cdf[symbol + 1]) // cdf[-1] - 1
        self.low = self.low + (range_size * cdf[symbol]) // cdf[-1]
        
        # Renormalize and output bits
        self._renormalize()
        
    def _renormalize(self):
        """Output bits when we can determine them."""
        while True:
            if self.high < self.HALF:
                # Entire interval is in [0, 0.5): output 0
                self._output_bit_plus_pending(0)
            elif self.low >= self.HALF:
                # Entire interval is in [0.5, 1): output 1
                self._output_bit_plus_pending(1)
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                # Interval straddles middle [0.25, 0.75): can't output yet
                self.pending_bits += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                # Can't determine any bits yet
                break
                
            # Scale up the interval
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
            
    def _output_bit_plus_pending(self, bit: int):
        """Output a bit followed by any pending opposite bits."""
        self.output_bits.append(bit)
        # Output pending bits (opposite of the bit we just output)
        opposite = 1 - bit
        for _ in range(self.pending_bits):
            self.output_bits.append(opposite)
        self.pending_bits = 0
        
    def finish(self) -> bytes:
        """
        Finish encoding and return compressed bytes.
        
        Must be called after encoding all symbols.
        """
        # Output enough bits to uniquely identify the final interval
        self.pending_bits += 1
        if self.low < self.QUARTER:
            self._output_bit_plus_pending(0)
        else:
            self._output_bit_plus_pending(1)
            
        # Pad to byte boundary
        while len(self.output_bits) % 8 != 0:
            self.output_bits.append(0)
            
        # Pack bits into bytes
        result = bytearray()
        for i in range(0, len(self.output_bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.output_bits[i + j]
            result.append(byte)
            
        return bytes(result)
    
    def get_num_bits(self) -> int:
        """Get current number of output bits (before finishing)."""
        return len(self.output_bits) + self.pending_bits


class ArithmeticDecoder(ArithmeticCoder):
    """Decodes a sequence of symbols given their probability distributions."""
    
    def __init__(self, data: bytes, precision_bits: int = 16):
        """
        Args:
            data: Compressed bytes from ArithmeticEncoder.finish()
            precision_bits: Must match encoder's precision_bits
        """
        super().__init__(precision_bits)
        
        # Convert bytes to bit stream
        self.bits: List[int] = []
        for byte in data:
            for i in range(7, -1, -1):
                self.bits.append((byte >> i) & 1)
        self.bit_pos = 0
        
        # Initialize state
        self.low = 0
        self.high = self.WHOLE
        
        # Read initial bits into value
        self.value = 0
        for _ in range(self.PRECISION):
            self.value = (self.value << 1) | self._read_bit()
            
    def _read_bit(self) -> int:
        """Read next bit from input, returning 0 if exhausted."""
        if self.bit_pos < len(self.bits):
            bit = self.bits[self.bit_pos]
            self.bit_pos += 1
            return bit
        return 0
        
    def decode_symbol(self, cdf: np.ndarray) -> int:
        """
        Decode a single symbol.
        
        Args:
            cdf: Cumulative distribution function (must match what encoder used)
            
        Returns:
            The decoded symbol
        """
        range_size = self.high - self.low + 1
        
        # Find which symbol interval contains our value
        # value is in [low + range * cdf[sym] / total, low + range * cdf[sym+1] / total)
        offset = self.value - self.low
        scaled_value = ((offset + 1) * cdf[-1] - 1) // range_size
        
        # Binary search for the symbol
        # Find largest i where cdf[i] <= scaled_value
        symbol = int(np.searchsorted(cdf, scaled_value, side='right')) - 1
        symbol = max(0, min(symbol, len(cdf) - 2))
        
        # Update interval (same as encoder)
        self.high = self.low + (range_size * cdf[symbol + 1]) // cdf[-1] - 1
        self.low = self.low + (range_size * cdf[symbol]) // cdf[-1]
        
        # Renormalize (same as encoder, but read bits instead of write)
        self._renormalize()
        
        return symbol
        
    def _renormalize(self):
        """Renormalize by reading new bits."""
        while True:
            if self.high < self.HALF:
                # Interval in lower half
                pass
            elif self.low >= self.HALF:
                # Interval in upper half
                self.value -= self.HALF
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                # Interval straddles middle
                self.value -= self.QUARTER
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
                
            # Scale up and read new bit
            self.low = 2 * self.low
            self.high = 2 * self.high + 1
            self.value = 2 * self.value + self._read_bit()


# =============================================================================
# Testing
# =============================================================================

def test_arithmetic_coding():
    """Test that encoding and decoding are inverse operations."""
    print("Testing arithmetic coding...")
    
    np.random.seed(42)
    vocab_size = 1024
    seq_length = 1000
    
    # Generate random probability distributions and symbols
    symbols = []
    cdfs = []
    
    for _ in range(seq_length):
        # Random probability distribution
        probs = np.random.dirichlet(np.ones(vocab_size))
        
        # Sample a symbol from this distribution
        symbol = np.random.choice(vocab_size, p=probs)
        symbols.append(symbol)
        
        # Convert to CDF
        coder = ArithmeticCoder()
        cdf = coder.probs_to_cdf(probs)
        cdfs.append(cdf)
    
    # Encode
    encoder = ArithmeticEncoder()
    for symbol, cdf in zip(symbols, cdfs):
        encoder.encode_symbol(symbol, cdf)
    compressed = encoder.finish()
    
    # Decode
    decoder = ArithmeticDecoder(compressed)
    decoded_symbols = []
    for cdf in cdfs:
        symbol = decoder.decode_symbol(cdf)
        decoded_symbols.append(symbol)
    
    # Verify
    assert symbols == decoded_symbols, "Decoded symbols don't match!"
    
    # Report compression
    original_bits = seq_length * 10  # 10 bits per token
    compressed_bits = len(compressed) * 8
    ratio = original_bits / compressed_bits
    
    print(f"  Sequence length: {seq_length}")
    print(f"  Original: {original_bits} bits")
    print(f"  Compressed: {compressed_bits} bits")
    print(f"  Ratio: {ratio:.2f}x")
    print("  ✓ Test passed!")
    
    return True


if __name__ == "__main__":
    test_arithmetic_coding()
