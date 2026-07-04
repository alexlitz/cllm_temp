import os
def total_units(mp):
    os.environ.pop('C4_MUL_MULTIPASS',None)
    if mp: os.environ['C4_MUL_MULTIPASS']='1'
    os.environ['C4_VM_CACHE_DIR']='/tmp/c4cache_units_'+('on' if mp else 'off')
    from importlib import reload
    import neural_vm.unified_compiler.full_vm_compiler_dynamic as F
    reload(F)
    model, layout = F.compile_full_vm_dynamic(disk_cache=False)
    tot=0; mul_partial=None; mul_combine=None
    import torch
    dp=layout.dim_positions
    for i,blk in enumerate(model.blocks):
        ffn=getattr(blk,'ffn',None)
        h=0
        # count units in PureFFN or composite
        def count(f):
            if f is None: return 0
            if hasattr(f,'W_up') and f.W_up is not None: return int(f.W_up.shape[0])
            if hasattr(f,'pipeline') and f.pipeline is not None:
                return sum(int(p.W_up.shape[0]) for p in f.pipeline if hasattr(p,'W_up') and p.W_up is not None)
            if hasattr(f,'inner'): return count(f.inner)
            return 0
        h=count(ffn); tot+=h
    return tot
off=total_units(False)
print("FLAG-OFF total FFN units:", off)
