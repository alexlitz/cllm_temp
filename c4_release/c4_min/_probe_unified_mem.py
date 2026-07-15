"""Prove the KV-memory head folded into the unified model's block-0 attention
is byte-exact: drive a store/load token stream through the ONE unified model's
own block-0 attention (softmax1 + ALiBi) and read the loaded value out of AX."""
import torch
import torch.nn.functional as F

from c4_min import blogspec_vocab as V
from c4_min.nibble_unified import build_unified_model
from c4_min.blogspec_memory import address_bits, _decode_byte
from c4_min.blogspec_layout import NIB_PER_REG


def run_mem(model, L, ops):
    """ops: list of ('store', addr, val) / ('load', addr). Drives block-0 attn
    of the UNIFIED model over the whole token stream; returns loaded values."""
    stream = [(V.BOS, {})]
    loaded = []
    for op in ops:
        if op[0] == "store":
            _, addr, val = op
            ov = {L.IS_STORE: 1.0}
            for b, bit in enumerate(address_bits(addr)):
                ov[L.ADDR_BIN + b] = bit
            for j, nv in enumerate(V.nibbles_of_value(val & 0xFFFFFFFF, NIB_PER_REG)):
                ov[L.VAL_NIB + j] = float(nv)
            stream.append((V.MEM, ov))
        else:  # load
            _, addr = op
            ov = {L.IS_LOAD: 1.0}
            for b, bit in enumerate(address_bits(addr)):
                ov[L.QRY_BIN + b] = bit
            stream.append((V.MEM, ov))
            toks = torch.tensor([[t for t, _ in stream]])
            with torch.no_grad():
                x = model.embed[toks].clone()
                for i, (_, o) in enumerate(stream):
                    for d, vv in o.items():
                        x[0, i, d] = vv
                # ONLY block-0 attention (the folded §Memory CAM head).
                x0 = model.blocks[0].attn(x)
            state = x0[0, -1]
            val = 0
            for bi in range(4):
                val |= _decode_byte(state, L, L.AX, bi) << (8 * bi)
            loaded.append(val)
            stream.pop()
    return loaded


if __name__ == "__main__":
    model, L, meta = build_unified_model(code_size=16, include_mdm_table=False)
    cases = [
        ([("store", 0x200, 42), ("load", 0x200)], [42]),
        ([("store", 0x200, 42), ("store", 0x204, 0xABCD), ("load", 0x204)], [0xABCD]),
        ([("store", 0x200, 42), ("load", 0x300)], [0]),               # ZFOD
        ([("store", 0x200, 42), ("store", 0x200, 99), ("load", 0x200)], [99]),  # latest
        ([("store", 0x200, 42), ("store", 0x200, 0), ("load", 0x200)], [0]),    # free
    ]
    allok = True
    for ops, exp in cases:
        got = run_mem(model, L, ops)
        ok = got == exp
        allok = allok and ok
        print(f"  {ops} -> {got} (exp {exp}) {'OK' if ok else 'FAIL'}")
    print("KV-MEMORY (folded in unified model block-0 attn) all ok:", allok)
