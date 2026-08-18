import sys, os
sys.path.insert(0,'/home/alexlitz/Documents/misc/c4_release/.claude/worktrees/task853-compiler-cfm/c4_release')
sys.path.insert(0,'/home/alexlitz/Documents/misc/c4_doom')
os.environ.setdefault('C4_DRAFT_READ_TO_MEM','1')
from src.compiler import compile_c
from c4_min import isa
from c4_min import nibble_filesys as FS
from c4_min.run_1096_pure_forward import bytecode_to_isa
from run_c4_min import tag_compiler_syscalls, install_compiler_abi_file_dispatcher, data_segment
from c4_min.pf_speculative import draft_pf_program

srcfile = sys.argv[1] if len(sys.argv)>1 else '/tmp/t853/minic.c'
inp = open(sys.argv[2],'rb').read() if len(sys.argv)>2 else b'int f(){ return 2+3*4; }'
src=open(srcfile).read()
bc,data=compile_c(src)
code=tag_compiler_syscalls(bytecode_to_isa(bc),isa)
data_seg=data_segment(data)
install_compiler_abi_file_dispatcher()
fio=FS.FileOpState(runner=FS.FileRunner(fs=FS.StubFilesystem({}), stdin=FS.InputKVStream(inp, neural=False)))
draft=draft_pf_program(code, max_steps=2000000, mask=0xFFFFFFFF, data_seg=data_seg, fio=fio)
out=bytes(fio.runner.stdout)
print('minic instrs=',len(code),'draft steps=',draft.step_count,'halted=',draft.halted)
print('OUTPUT (%d bytes):'%len(out))
sys.stdout.write(out.decode('latin1'))
sys.stdout.flush()
