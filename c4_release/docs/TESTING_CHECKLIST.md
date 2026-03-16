
Testing checklist


- All of the 1000+ comprehensive tests work
- The network is 100% autoregressive, not using any external memory or logic only using standard layers
- It is able to export and run via onnx, still passing the 100+ tests
- IO behavior with 100% pure autoregressive transformer works with the reading and writing user messages
- Tool use IO works correctly
- KV cache eviction works properly and maintains correct outputs over even long problems
- Running through the onnx runtime in c4 c works and passes the 1000+ tests
- The bundler, bundles programs, the model weights and the program bytecode all together into a single file which runs the bytecode via executing the bundled model in onnx runtime, and passes the 1000+ tests. A version of the bundler written in C4 C should also exist and pass all 1000+ tests
- The quine, a program which outputs its own source code, runs correctly and passes the 1000+ tests. Should be written in c4 c, run via the model, include the runtime, the model weights and program bytecode.
- The network structure, it should be 100% a vanilla transformer using MoE, SwiGLU, vanilla attention. None of the operations should be performed in any other way, such as via external memory or custom non-transformer layers. The network should be able to be exported to onnx and run in onnx runtime, still passing all 1000+ tests.
