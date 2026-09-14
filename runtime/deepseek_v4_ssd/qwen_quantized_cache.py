"""Optional QSA storage quantization; DeltaNet recurrent state stays unmodified."""
import mlx.core as mx
from mlx_lm.models.cache import QuantizedKVCache


class QSAQuantizedCache(QuantizedKVCache):
    def __init__(self, bits, dimension):
        if bits not in (4, 8) or dimension % 32:
            raise ValueError('QSA quantized cache requires 4 or 8 bits and a width divisible by 32')
        super().__init__(group_size=32, bits=bits)
        self.dimension = dimension

    def update_and_fetch(self, keys, values):
        if keys.shape[-1] != self.dimension or values.shape[-1] != self.dimension:
            raise ValueError('QSA cache width does not match')
        packed = super().update_and_fetch(keys, values)
        # QSA selects sparse micro-blocks using the existing attention path.
        # Only these temporary views are dequantized; retained history stays packed.
        return tuple(mx.dequantize(*item, group_size=self.group_size, bits=self.bits)
                     for item in packed)

    def persistence_state(self):
        return dict(bits=self.bits, dimension=self.dimension, offset=self.offset,
                    arrays=super().state if self.keys is not None else None)

    def restore_persistence_state(self, saved):
        if (not isinstance(saved, dict) or set(saved) != {'bits', 'dimension', 'offset', 'arrays'}
                or saved['bits'] != self.bits or saved['dimension'] != self.dimension
                or type(saved['offset']) is not int or saved['offset'] < 0):
            raise ValueError('Invalid QSA packed cache format')
        arrays, offset = saved['arrays'], saved['offset']
        if arrays is None:
            if offset:
                raise ValueError('Empty QSA cache has nonzero offset')
            self.keys = self.values = None
            self.offset = 0
            return
        if not isinstance(arrays, (list, tuple)) or len(arrays) != 2:
            raise ValueError('Invalid QSA packed cache arrays')
        first = None
        for item in arrays:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError('Invalid QSA packed cache component')
            for i, array in enumerate(item):
                width = self.dimension * self.bits // 32 if i == 0 else self.dimension // 32
                if (not isinstance(array, mx.array) or array.ndim != 4 or array.shape[0] != 1
                        or array.shape[2:] != (offset, width)
                        or (i == 0 and array.dtype != mx.uint32)
                        or (i > 0 and array.dtype not in (mx.bfloat16, mx.float16, mx.float32))):
                    raise ValueError('Invalid QSA packed cache shape or dtype')
                shape = array.shape[:3]
                if first is None:
                    first = shape
                if shape != first:
                    raise ValueError('QSA packed cache components do not match')
        self.keys, self.values = [tuple(mx.array(a) for a in item) for item in arrays]
        self.offset = offset
