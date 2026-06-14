import os,sys
_H=os.path.dirname(os.path.abspath(__file__)); _P=os.path.dirname(_H); sys.path.insert(0,_P)
os.environ["C4_SMOKE_SPEC_K"]="0"; os.environ["C4_TEST_SPEC_K"]="0"
import contextlib,io
with contextlib.redirect_stderr(io.StringIO()):
    from neural_vm.run_vm import AutoregressiveVMRunner
    from neural_vm.batched_pure_neural import BatchedPureNeuralRunner, Token
    from tests.test_suite_1000 import generate_test_programs
    from src.compiler import compile_c
    from tests.declarative_oracle import declarative_oracle_for_program
    mr=AutoregressiveVMRunner(pure_neural=True, trust_neural_alu=True, spec_k=0)
    mr._func_call_handlers={}; mr._syscall_handlers={}
    runner=BatchedPureNeuralRunner(model_runner=mr)
for pid in [int(x) for x in sys.argv[1:]]:
    src,exp,desc=generate_test_programs()[pid]; bc=compile_c(src)[0]
    orc=declarative_oracle_for_program(list(bc), b'', label='p')
    res=runner.run_batch([list(bc)], max_steps=None, spec_k=0, expected_steps_list=[orc.steps], bucket_by_predicted_length=False)[0]
    print(f"RESULT id{pid} exp={exp} spec0_result={res} u={runner.model.blocks[46].ffn.W_down.shape[1]}", flush=True)
