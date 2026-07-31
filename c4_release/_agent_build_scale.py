import warnings, time, torch, sys
warnings.filterwarnings("ignore")
from c4_min.lib_neural import build_lib_model_streaming
for cs in [64, 256, 512, 1024, 2048]:
    t0=time.time()
    try:
        model,L,stats=build_lib_model_streaming(code_size=cs, recurrent_divmod=True, addr32=True, compute_mode="dense_kernel")
        nb=len(model.blocks)
        d=getattr(L,'dim',None) or getattr(model.blocks[0].attn,'head_dim',None)
        print(f"code_size={cs}: build={time.time()-t0:.1f}s blocks={nb} hidden~{getattr(L,'dim','?')}", flush=True)
        del model
    except Exception as e:
        print(f"code_size={cs}: ERROR {type(e).__name__} {e} (after {time.time()-t0:.1f}s)", flush=True)
        break
