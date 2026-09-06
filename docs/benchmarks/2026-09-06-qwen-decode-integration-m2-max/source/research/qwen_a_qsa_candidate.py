"""Default-off QSA candidate for research runners only.

Keep selected indices, causal mask and 2-head K/V gather unchanged, replace only
attention math by MLX's fused grouped-query SDPA. This follows the IO-aware
attention mechanism in https://arxiv.org/abs/2205.14135; fused rounding differs
from the production BF16 intermediate path and must pass a full-model gate.
"""
import inspect
import textwrap

import mlx.core as mx
from deepseek_v4_ssd import qwen4_exp as qwen


def build_sdpa():
    source = textwrap.dedent(inspect.getsource(qwen.QSAAttention._bounded_attention))
    start = source.index("        grouped_query = current_query.reshape(")
    end = source.index("        outputs.append(", start)
    source = source[:start] + '''        current = mx.fast.scaled_dot_product_attention(
            current_query[:, :, None, :], selected_key, selected_value,
            scale=self.args.head_dim**-0.5,
            mask=causal[:, None, None, :],
        ).squeeze(-2)
''' + source[end:]
    namespace = dict(vars(qwen))
    exec(compile(source, "<qwen-qsa-sdpa-research>", "exec"), namespace)
    return namespace["_bounded_attention"]


def build_grouped_tile(tile):
    """Retile the post-adoption Grouped-KV implementation, same operations.

    The rejected September 2 chunk-8 gate preceded removal of 12-fold KV
    expansion. This candidate keeps the two KV heads unexpanded.
    """
    if tile not in (16, 64):
        raise ValueError("research tile must be 16 or 64")
    source = textwrap.dedent(inspect.getsource(qwen.QSAAttention._bounded_attention))
    if source.count("query_chunk = 4") != 1:
        raise ValueError("QSA source contract changed")
    source = source.replace("query_chunk = 4", f"query_chunk = {tile}")
    namespace = dict(vars(qwen))
    exec(compile(source, "<qwen-grouped-kv-retile-research>", "exec"), namespace)
    return namespace["_bounded_attention"]


def build_direct():
    """Fuse selected-row loads with grouped QK and PV, preserving BF16 stages."""
    qk = mx.fast.metal_kernel(
        name="qwen_direct_grouped_qk",
        input_names=["query","key","selected","valid"],
        output_names=["scores"],
        source=r'''
        uint lane = thread_index_in_simdgroup;
        uint s = thread_position_in_grid.y;
        uint n = thread_position_in_grid.z;
        if (s >= NS || n >= QL*HK) return;
        uint q = n / HK, hk = n % HK;
        float dots[R];
        for (uint r=0;r<R;++r) dots[r]=0.0f;
        if (valid[q*NS+s]) {
            uint index=selected[q*NS+s];
            for (uint d=lane;d<D;d+=32) {
                float kval=float(key[(hk*KL+index)*D+d]);
                for (uint r=0;r<R;++r)
                    dots[r]+=float(query[((hk*R+r)*QL+q)*D+d])*kval;
            }
        }
        for (uint r=0;r<R;++r) {
            float dot=simd_sum(dots[r]);
            if (lane==0) {
                InT rounded=InT(dot);
                scores[(q*HQ+hk*R+r)*NS+s] = valid[q*NS+s]
                    ? InT(float(rounded)*metal::rsqrt(float(D))) : InT(-3.3895313892515355e38f);
            }
        }
        ''')
    pv = mx.fast.metal_kernel(
        name="qwen_direct_grouped_pv",
        input_names=["weights","value","selected"],
        output_names=["output"],
        source=r'''
        uint lane=thread_index_in_simdgroup;
        uint d=thread_position_in_grid.y;
        uint n=thread_position_in_grid.z;
        if (d >= D || n >= QL*HK) return;
        uint q=n/HK,hk=n%HK;
        float sums[R];
        for (uint r=0;r<R;++r) sums[r]=0.0f;
        for (uint s=lane;s<NS;s+=32) {
            uint index=selected[q*NS+s];
            float val=float(value[(hk*KL+index)*D+d]);
            for (uint r=0;r<R;++r)
                sums[r]+=float(weights[(q*HQ+hk*R+r)*NS+s])*val;
        }
        for (uint r=0;r<R;++r) {
            float total=simd_sum(sums[r]);
            if (lane==0) output[((hk*R+r)*QL+q)*D+d]=InT(total);
        }
        ''')

    def direct(query,key,value,selected,valid):
        _,hq,ql,d=query.shape
        hk,kl=key.shape[1:3]
        ns=selected.shape[1]
        template=[("InT",query.dtype),("HQ",hq),("HK",hk),("QL",ql),
            ("KL",kl),("NS",ns),("D",d),("R",hq//hk)]
        scores=qk(inputs=[query,key,selected,valid],template=template,
            grid=(32,ns,ql*hk),threadgroup=(32,4,1),
            output_shapes=[(ql,hq,ns)],output_dtypes=[query.dtype])[0]
        weights=mx.softmax(scores.astype(mx.float32),axis=-1).astype(query.dtype)
        return pv(inputs=[weights,value,selected],template=template,
            grid=(32,d,ql*hk),threadgroup=(32,4,1),
            output_shapes=[query.shape],output_dtypes=[query.dtype])[0]

    source=textwrap.dedent(inspect.getsource(qwen.QSAAttention._bounded_attention))
    source=source.replace("    outputs = []", "    selections, validity = [], []")
    start=source.index("        selected_key = mx.take(")
    end=source.index("    return mx.concatenate(outputs, axis=2)",start)
    source=source[:start]+'''        selections.append(selected)
        validity.append(selected_valid & (selected <= absolute[:, None]))
    return _direct(query, key, value, mx.concatenate(selections, axis=0), mx.concatenate(validity, axis=0))
'''
    namespace={**vars(qwen),"_direct":direct}
    exec(compile(source,"<qwen-direct-selected-attention>","exec"),namespace)
    return namespace["_bounded_attention"]


def install_prefill_sdpa():
    original = qwen.QSAAttention._bounded_attention
    candidate = build_sdpa()

    def dispatch(self, query, *args, **kwargs):
        implementation = candidate if query.shape[2] > 1 else original
        return implementation(self, query, *args, **kwargs)

    qwen.QSAAttention._bounded_attention = dispatch
