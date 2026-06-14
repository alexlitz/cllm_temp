import os,sys
_H=os.path.dirname(os.path.abspath(__file__)); _P=os.path.dirname(_H); sys.path.insert(0,_P)
import contextlib,io
with contextlib.redirect_stderr(io.StringIO()):
    from neural_vm.unified_compiler.full_vm_compiler_dynamic import compile_full_vm_dynamic
    model,layout=compile_full_vm_dynamic(disk_cache=True, alu_mode="efficient")
print(f"RESULT block46_units={model.blocks[46].ffn.W_down.shape[1]}", flush=True)
