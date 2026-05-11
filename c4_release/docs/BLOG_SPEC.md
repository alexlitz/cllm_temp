## Introduction/background

So I wanted to have a C compiler that compiles to a transformer. That is C code in and transformer weights out, weights which execute the C program and have the same input output behavior as the corresponding C program. I want it to be as much as possible a 100% pure transformer with no exotic architecture choices hacks only being clever not stretching the definitions, someone should be able to look at the forward or the onnx and say yup that is a standard transformer nothing much strange going on there. No encoder-decoder no python loop other than the standard generation loop, no auxiliary memory or python variables, no special memory management or special masking, just a standard auto regressive decode only transformer and generation loop and things that are actually done in real LLMs for optimizations. I gear the implementation towards being vanilla and that involves a fair amount of personal taste. E.G. tool calling is increasingly common but I consider it not a part of the LLM itself, and after all if you allowed it you may as well implement this using a compiler as a tool. Other things would make it too simple, too easy. I aim to make this a very comprehensive rundown of what this is and how I did it.
## Why

Because it would be cool, because it would show that transformers can do exact computations in form, because it will show a way how each of these computations can be done, because if you extended it to optimizing compilers you might have an alternative to training, because you could embed these functionalities into larger transformers to perform operations perfectly without tool calls and thus more efficiently, because it is a nice shortcut to getting a "transformer that can do X" to be able to just write a C program to do it and get one. It can be educating for me and thee going through the exercise. Because it is fun! Because it is perfectly explainable AI -- if you count it as AI. Everybody wants to move everything to LLMs and generative AI and so we of course need a solution for how we can do that for things that are more naturally C programs, like cli utilities like cat, echo, yes etc.. And of course because somebody had to!

## An Elephant in the Room

Hilariously I got scooped in writing/making this. 

So if you are reading this there is a decent chance you are thinking what another?? And "maybe of course someone had to but someone did!" I started this on March 3 after having had fun making and writing [[personal/blog_posts/published/Building a Minimal Transformer for 10-digit Addition|Building a Minimal Transformer for 10-digit Addition]] on March 11th after having finished most of the code and a substantial amount of this writeup I saw [Can LLMs be Computers](https://www.percepta.ai/blog/can-llms-be-computers) damn well still gotta finish what I started and so I have.

So naturally I need to lay out some selling points:
- It is open source you can checkout the code [here](https://github.com/alexlitz/c4llm) and compile your own C programs to a transformer
- I tell you in pretty good detail about every single weight and what it does
- I keep things almost entirely vanilla as far as the transformer architecture
- Self-hosting and Quine are pretty neat
- You like reading technical blogposts

Take that for what you will and let's get started!

## C4

C4 is a minimalist C compiler written in ~500 lines of C. It consists of four main functions:
1. **next()** - Lexical analyzer that tokenizes input
2. **expr()** - Expression parser using recursive descent
3. **stmt()** - Statement parser
4. **main()** - Entry point with VM execution loop

C4 supports a subset of C including:
- Basic types: int, char, pointers
- Control flow: if/else, while, return
- Operators: arithmetic, bitwise, comparison
- Functions with recursion
- Pointers and arrays
- Basic I/O via printf, getchar

The bytecode format uses variable-length instructions where opcodes 0-8 (stack/address operations) are followed by an immediate operand. Nice and simple, it works by parsing the code -- a subset of C that is supported into bytecode for a simple VM with about 40 operations. So if we can support those operations we can execute that bytecode. But we don't want bytecode input we want C code input, easy enough, C4 can self host so we can simply implement the virtual machine in the transformer, give it the c4 program to execute as a system prompt, have the c4 program parse the user input program into bytecode then pass the control flow for the transformer VM to execute the user program bytecode. We also want the user output to be just what would be in stdout/stderr, but the transformer is doing the VM execution autoregressivly internally, thankfully internal thinking being separated from user facing output has been solved in what is now mainstream of LLMs, just put that stuff in a \<think\> tag. So this gets us pretty far, but we don't want a transformer and system prompt that is able to execute the C program we want just a transformer that is able to execute it. This is easy enough, we can substitute operations using attention on a fixed prompt with equivalent operations in the FFN to effectively bake in a system prompt. So we bake in the C4 bytecode to the system prompt and voilà we have a transformer that is executing C code on its' own. Don't say I buried the lead... But of course it is a bit more complicated than that and hence this is a longer post.

However this being a transformer we have very different primitives to work with than C4 does. C4 being written in the C4 subset of C it is able to very straightforwardly implement the operations needed to execute the VM. In a transformer we don't have that luxury. This makes a line by line reimplementation effectively impossible. We need to map all of the operations onto a transformer some are more straightforward than others. Some like multiplication can use as few as 6 weights to do the general operation but require more to do without assuming high precision floats, requiring it to meet the int32 spec. Some simple things like keeping track of the registers require a surprising amount of computation if not weights.

### C4 Registers

| Name          |           Size | Description                                           |                    Initial Value |
| ------------- | -------------: | ----------------------------------------------------- | -------------------------------: |
| PC            |         32-bit | **Program Counter** — address of the next instruction |                                0 |
| AX            |         32-bit | **Accumulator** — main working register for results   |                                0 |
| SP            |         32-bit | **Stack Pointer** — top of stack                      |              0x10000 (or 0x8000) |
| BP            |         32-bit | **Base Pointer** — base of current stack frame        |              0x10000 (or 0x8000) |
| **Immediate** | 32-bit operand | Literal constant embedded in the instruction stream   | Read from code at PC when needed |

### C4 Opcodes and How They Are Implemented In C4
|                 # | Name | Description                      |
| ----------------: | ---- | -------------------------------- |
| **Stack/Address** |      |                                  |
|                 0 | LEA  | AX = BP + imm                    |
|                 1 | IMM  | AX = imm                         |
|                 2 | JMP  | PC = imm                         |
|                 3 | JSR  | push PC, PC = imm                |
|                 4 | BZ   | if AX==0: PC = imm               |
|                 5 | BNZ  | if AX!=0: PC = imm               |
|                 6 | ENT  | push BP, BP=SP, SP-=imm          |
|                 7 | ADJ  | SP += imm                        |
|                 8 | LEV  | SP=BP, pop BP, pop PC            |
|        **Memory** |      |                                  |
|                 9 | LI   | AX = *AX (KV attention)          |
|                10 | LC   | AX = _(char_)AX                  |
|                11 | SI   | *pop = AX (KV write)             |
|                12 | SC   | _(char_)pop = AX                 |
|                13 | PSH  | push AX                          |
|       **Bitwise** |      |                                  |
|                14 | OR   | AX = pop \| AX                   |
|                15 | XOR  | AX = pop ^ AX                    |
|                16 | AND  | AX = pop & AX                    |
|    **Comparison** |      |                                  |
|                17 | EQ   | AX = (pop == AX)                 |
|                18 | NE   | AX = (pop != AX)                 |
|                19 | LT   | AX = (pop < AX)                  |
|                20 | GT   | AX = (pop > AX)                  |
|                21 | LE   | AX = (pop <= AX)                 |
|                22 | GE   | AX = (pop >= AX)                 |
|         **Shift** |      |                                  |
|                23 | SHL  | AX = pop << AX                   |
|                24 | SHR  | AX = pop >> AX                   |
|    **Arithmetic** |      |                                  |
|                25 | ADD  | AX = pop + AX                    |
|                26 | SUB  | AX = pop - AX                    |
|                27 | MUL  | AX = pop * AX                    |
|                28 | DIV  | AX = pop / AX                    |
|                29 | MOD  | AX = pop % AX                    |
|        **System** |      |                                  |
|                30 | OPEN | AX = open(file) via tool call    |
|                31 | READ | AX = read(fd,buf,n) via input KV |
|                32 | CLOS | close(fd) → 0                    |
|                33 | PRTF | printf(fmt,...)                  |
|                34 | MALC | AX = malloc(n) bump alloc        |
|                35 | FREE | free(ptr) zero overwrite         |
|                36 | MSET | memset(p,v,n) loop               |
|                37 | MCMP | memcmp(a,b,n) loop               |
|                38 | EXIT | exit(code)                       |
|       **Control** |      |                                  |
|                39 | NOP  | no-op                            |
|                40 | POP  | SP += 8                          |

|                 # | Name    |   L |   W | Description                      |
| ----------------: | ------- | --: | --: | -------------------------------- |
| **Stack/Address** |         |     |     |                                  |
|                 0 | LEA     |   1 |  20 | AX = BP + imm                    |
|                 1 | IMM     |   0 |   0 | AX = imm                         |
|                 2 | JMP     |   1 |  12 | PC = imm                         |
|                 3 | JSR     |   2 |  30 | push PC, PC = imm                |
|                 4 | BZ      |   1 |  10 | if AX==0: PC = imm               |
|                 5 | BNZ     |   1 |  10 | if AX!=0: PC = imm               |
|                 6 | ENT     |   2 |  40 | push BP, BP=SP, SP-=imm          |
|                 7 | ADJ     |   1 |  16 | SP += imm                        |
|                 8 | LEV     |   2 |  40 | SP=BP, pop BP, pop PC            |
|        **Memory** |         |     |     |                                  |
|                 9 | LI      |   1 | 500 | AX = *AX (KV attention)          |
|                10 | LC      |   1 | 500 | AX = *(char*)AX                  |
|                11 | SI      |   1 | 500 | *pop = AX (KV write)             |
|                12 | SC      |   1 | 500 | *(char*)pop = AX                 |
|                13 | PSH     |   2 |  80 | push AX                          |
|       **Bitwise** |         |     |     |                                  |
|                14 | OR      |   1 |  80 | AX = pop \| AX                   |
|                15 | XOR     |   1 |  80 | AX = pop ^ AX                    |
|                16 | AND     |   1 |  80 | AX = pop & AX                    |
|    **Comparison** |         |     |     |                                  |
|                17 | EQ      |   1 |  80 | AX = (pop == AX)                 |
|                18 | NE      |   1 |  80 | AX = (pop != AX)                 |
|                19 | LT      |   2 | 100 | AX = (pop < AX)                  |
|                20 | GT      |   2 | 100 | AX = (pop > AX)                  |
|                21 | LE      |   2 | 110 | AX = (pop <= AX)                 |
|                22 | GE      |   2 | 110 | AX = (pop >= AX)                 |
|         **Shift** |         |     |     |                                  |
|                23 | SHL     |   2 | 160 | AX = pop << AX                   |
|                24 | SHR     |   2 | 160 | AX = pop >> AX                   |
|    **Arithmetic** |         |     |     |                                  |
|                25 | ADD     |   2 | 120 | AX = pop + AX                    |
|                26 | SUB     |   2 | 120 | AX = pop - AX                    |
|                27 | MUL     |   4 | 320 | AX = pop * AX                    |
|                28 | DIV     |   6 | 250 | AX = pop / AX                    |
|                29 | MOD     |   7 | 370 | AX = pop % AX                    |
|        **System** |         |     |     |                                  |
|                30 | OPEN    |   1 |  20 | AX = open(file) via tool call    |
|                31 | READ    |   9 | 250 | AX = read(fd,buf,n) via input KV |
|                32 | CLOS    |   1 |  10 | close(fd) → 0                    |
|                33 | PRTF    |   1 |  20 | printf(fmt,...)                  |
|                34 | MALC    |   1 |  51 | AX = malloc(n) bump alloc        |
|                35 | FREE    |   1 |  27 | free(ptr) zero overwrite         |
|                36 | MSET    |   1 |  16 | memset(p,v,n) loop               |
|                37 | MCMP    |   1 |  16 | memcmp(a,b,n) loop               |
|                38 | EXIT    |   1 |   9 | exit(code)                       |
|       **Control** |         |     |     |                                  |
|                39 | NOP     |   0 |   0 | no-op                            |
|                40 | POP     |   1 |  16 | SP += 8                          |
|                41 | BLT     |   1 |  17 | branch if signed <               |
|                42 | BGE     |   1 |  17 | branch if signed >=              |
|           **I/O** |         |     |     |                                  |
|                64 | GETCHAR |   9 | 220 | AX = getchar()                   |
|                65 | PUTCHAR |   9 | 220 | putchar(AX)                      |
  

**L** = Layers, **W** = Weights

---

  

## Summary

  

| Category | Count | Max L | Weights |

|----------|------:|------:|--------:|

| Stack/Address | 9 | 2 | 178 |

| Memory | 5 | 2 | 2,080 |

| Bitwise | 3 | 1 | 240 |

| Comparison | 6 | 2 | 580 |

| Shift | 2 | 2 | 320 |

| Arithmetic | 5 | 7 | 1,180 |

| System | 9 | 9 | 419 |

| Control | 4 | 1 | 50 |

| I/O | 2 | 9 | 440 |

| **Total** | **45** | **9** | **5,487** |


Redundancy:


  | Sharing Type       | Reported | Unique | Factor | Examples                   |
  |--------------------|----------|--------|--------|----------------------------|
  | 8 nibbles parallel | 400      | 50     | 8x     | OR, XOR, AND, EQ, NE       |
  | 8 cascade layers   | 930      | 150    | 6x     | GETCHAR, PUTCHAR, ADD, SUB |
  | 6 iterations       | 250      | 50     | 5x     | DIV (Newton-Raphson)       |
  | KV projections     | 2,000    | 400    | 5x     | LI, LC, SI, SC             |
  | No sharing         | 397      | 397    | 1x     | Stack ops, EXIT, etc.      |

  Total: 5,487 reported → 1,397 unique weights

  
  | Counting Method            | Tensor Size | Non-Zero |
  |----------------------------|-------------|----------|
  | Naive (all ops separate)   | 153,232     | 5,487    |
  | With nibble/layer sharing  | 153,232     | 1,397    |
  | With full op-class sharing | 88,548      | ~800     |

  The entire Neural VM needs only ~800 unique non-zero weights.

  Shared modules:
  - Memory KV (LI/LC/SI/SC): 100 weights — same Q/K/V projection
  - I/O Cascade: 35 weights — same nibble extraction × 8 layers
  - Nibble Arith: 20 weights — same sum+carry × 8 nibbles
  - Bitwise: 10 weights — one formula (a+b-ab) for all
  - Comparison: 40 weights — diff + cascade priority
  - Shift: 40 weights — nibble routing

  The tensors are 96.4% sparse — most entries are zero. The actual computation is controlled by ~800 carefully placed non-zero weights.
  
TODO format tables

## Vanillaness

I felt it was important to have this be as much a Vanilla transformer as possible. Otherwise there are a lot of hacks that can be pretty subtle and take the whole complexity/fun out of the affair. The only real deviation I have is using softmax1 which I like and think should be more standard and arguably using ALiBi. Softmax1 can be simulated by getting a token to realize it is special and act like a sink, everything I did with ALiBi could be done a different way with RoPE. So chalk it up to taste. Otherwise, standard attention and MoE SwiGLU FFNs alternating. They do vary in size between layers and experts but hey you can just imagine the zeros are there and it will work the same.

Here are the classes and forward functions everything else is 

```python
class PureFFN(nn.Module):
    """
    Pure SwiGLU FFN with FINAL forward pass.

    Uses nn.Linear internally for standard PyTorch patterns while
    providing direct weight access for baking via W_up, W_gate, W_down properties.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        # Use nn.Linear with biases
        self.up = nn.Linear(dim, hidden_dim, bias=True)
        self.gate = nn.Linear(dim, hidden_dim, bias=True)
        self.down = nn.Linear(hidden_dim, dim, bias=True)

        # Zero initialize
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

        self._bake_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FINAL - Standard SwiGLU FFN. DO NOT OVERRIDE."""
        up = self.up(x)
        gate = self.gate(x)
        hidden = F.silu(up) * gate
        return x + self.down(hidden)
        
    def _bake_weights(self):
        """Override to bake operation-specific weights."""
        pass
```

TODO get these more cleaned up/tight

```python

class PureAttention(nn.Module):
    """Multi-head attention with softmax1 (ZFOD) and ALiBi positional bias.

    Uses nn.Linear internally. Subclasses override _bake_weights().
    """

    def __init__(self, dim: int, num_heads: int = 4, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len

        # Use nn.Linear
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Zero initialize
        # ...

        # ALiBi slopes: geometric sequence 2^(-8/n * (i+1)) for each head
        slopes = torch.tensor(
            [2.0 ** (-8.0 / num_heads * (i + 1)) for i in range(num_heads)]
        )
        self.register_buffer("alibi_slopes", slopes)

        self._bake_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        linear = sparse_linear if self.q_proj.weight.is_sparse else F.linear
        Q = linear(x, self.q_proj.weight).view(B, S, H, HD).transpose(1, 2)
        K = linear(x, self.k_proj.weight).view(B, S, H, HD).transpose(1, 2)
        V = linear(x, self.v_proj.weight).view(B, S, H, HD).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # ALiBi bias
        positions = torch.arange(S, device=x.device)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        alibi = -self.alibi_slopes.view(1, H, 1, 1) * dist
        scores = scores + alibi

        # Causal mask
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + causal_mask

        # softmax1 for ZFOD
        attn = softmax1(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        linear = sparse_linear if self.q_proj.weight.is_sparse else F.linear
        return x + linear(out, self.out_proj.weight)

    def _bake_weights(self):
        """Override to bake attention weights."""
        pass
```

```python
class SoftMoEFFN(nn.Module):
    """
    Soft Mixture-of-Experts FFN layer.

    All experts run in parallel. Outputs are blended by opcode weights.
    NO Python control flow in forward - pure tensor operations only.

    forward(x) = sum over all experts of: opcode_weight[expert_op] * expert(x)
    """

    def __init__(self, experts: List[PureFFN], expert_opcodes: List[int]):
        """
        Args:
            experts: List of PureFFN expert modules
            expert_opcodes: List of opcode numbers, one per expert
        """
        super().__init__()
        self.experts = nn.ModuleList(experts)
        # Store opcodes as Python list (not tensor) for ONNX compatibility
        self.expert_opcode_list = list(expert_opcodes)
        # Also store as buffer for state_dict
        self.register_buffer('expert_opcodes', torch.tensor(expert_opcodes, dtype=torch.long))
        self.num_experts = len(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure tensor forward - no Python control flow.

        All experts run, outputs weighted by opcode one-hot.
        ONNX compatible: uses Python ints (static at trace time).
        """
        # Get opcode weights from embedding [batch, pos, NUM_OPS]
        opcode_weights = x[:, 0, E.OP_START:E.OP_START + E.NUM_OPS]  # [batch, NUM_OPS]

        # Initialize output as zeros
        output = torch.zeros_like(x)

        # Run ALL experts and accumulate weighted outputs
        for i in range(self.num_experts):
            expert_out = self.experts[i](x)  # [batch, pos, dim]
            # Get weight for this expert's opcode (Python int, static at trace time)
            opcode_idx = self.expert_opcode_list[i]
            weight = opcode_weights[:, opcode_idx:opcode_idx+1].unsqueeze(-1)  # [batch, 1, 1]
            output = output + weight * (expert_out - x)  # Weighted residual

        return x + output
```
We combine these into MoEs TODO elaborate
We also want the generate function to be vanilla TODO generate function

Given a transformer is made of math and we are doing math we can pretty easily "cheat" to make the notional parameters lower or do operations more simply I think limiting things architecturally this way puts a check on that.

## Tokenization

One problem is tokenization, we want to operate on 32-bit values, but having $2^{32}$ is a lot of tokens and makes it a lot harder to do bitwise operations. So instead we use bytes and internally break them into nibbles (4-bits) for easy of use. It is pretty easy to go from bits to integers or floats but decently harder to go in the other direction. Because we want to keep track of the registers we write the up to date values each step. This means many of the passes of the transformer are just outputting the registers and not doing meaningful computation
## Memory

The program memory works by having a store instruction which attends to its registers to get the address and value it is storing, it represents the address in binary and sets the key to +scale for ones, -scale for zeros. Which allows us to retrieve this position in memory by attending with a query identical to the key. The scale should be large enough that other keys differing in one or more position have trivial weights, the positional bias should be such that earlier writes to the position have trivial weight and the sum of the $scale^2$ values should be large enough that the 1 in softmax1 is trivial even with extremely significant positional bias as will occur if the store is far in the past. At first I was thinking that it might be necessary to rewrite the full memory and have some fancy masking or special KV cache whenever a memory write was made but positional bias via AliBi to heavily upweight more recent items, such that it strictly prioritized exact address match and subject to that having priority to the most recently written store to the memory does exactly what we want. 

We need the multi-token stores in the text to correspond to one entry in the kv cache with the full value, so we need to basically keep track of where we are in the VM step and when we are at the right point, attend to the memory for the correct address and memory and set the key and value accordingly. At first glance this may seem counter intuitive, if we have the information to generate the tokens to generate the address and value surly we could just directly set it in the KV cache. The issue is that that information is or may be at a higher level then where we want it in the KV cache, so while it can be used to control generation we can't get it from A to B (or B to A as a lexicographical ordering of layers might have it) without first going through the tokens, so it is a bit of a kludge, this may account for some degree of the effectiveness of looped transformers, if you can flag that it needs a refined value in a future layer then that could substitute for several thinking tokens in cases like this, i.e. writes would have a flag and reads would read that flag and loop till the appropriate value is in the KV cache. De-facto all memory addresses are 4-byte aligned, all loads and writes are 4 bytes, non-aligned reads and writes are invalid and all addresses are helpfully initialized to zero by softmax1.

## How Bytecode is Passed to the Network


The System Prompt Format is as follows

| Section      | Contents                                                      |
| ------------ | ------------------------------------------------------------- |
| **BYTECODE** | 5 bytes per instruction: `[op:1] [imm:4 little-endian]`       |
| **SEP**      | Token `256`                                                   |
| **DATA**     | Raw bytes                                                     |
| **SEP**      | Token `256`                                                   |
| **ARGV**     | Null-terminated strings, e.g. `"program\0" "arg1\0" "arg2\0"` |

```text
0000: 01 06 00 00 00        IMM  6
0005: 0d 00 00 00 00        PSH
000a: 01 07 00 00 00        IMM  7
000f: 1b 00 00 00 00        MUL
0014: 26 00 00 00 00        EXIT
0019: SEP
001a: 48 65 6c 6c 6f 20 57 6f 72 6c 64   "Hello World"
0025: SEP
0026: 70 72 6f 67 72 61 6d 00            "program\0"
002e: 61 72 67 31 00                     "arg1\0"
0033: 61 72 67 32 00                     "arg2\0"
```

The system prompt is relatively straightforward, it gets the bytecode passed in as bytes as seperators, then argc and argv
## Registers

We want to keep track of register values as well, for a similar reason we want to write these auto-regressively, so we write them every step, these are also 4 tokens each, so we end up writing 30 tokens for each VM step

| Tokens | Content         | Purpose                         |
| ------ | --------------- | ------------------------------- |
| 1      | REG_PC marker   | Indicates PC register follows   |
| 4      | PC bytes        | Program counter (little-endian) |
| 1      | REG_AX marker   | Indicates AX register follows   |
| 4      | AX bytes        | Accumulator value               |
| 1      | REG_SP marker   | Indicates SP register follows   |
| 4      | SP bytes        | Stack pointer                   |
| 1      | REG_BP marker   | Indicates BP register follows   |
| 4      | BP bytes        | Base pointer                    |
| 1      | MEM marker      | Indicates memory operation      |
| 8      | Addr + Value    | Memory write (4 bytes each)     |
| 1      | STEP_END marker | Marks end of VM step            |
| **30** | **Total**       |                                 |

We could pretty easily avoid doing memory writing each VM step but it keeps it nice and simple to just do NULL memory writes.

## The Operations

### The FFN

We are using SwiGLU FFNs, which have three matrices, up, gate, down.

```python
def sigmoid(x):
    return exp(x) / (1 + exp(x))
    
def SiLU(x):
    return sigmoid(x) * x

def SwiGLU(up, gate, down, x):
    return down @ (SiLU(up @ x) * (gate @ x))
```

SiLU ends up looking pretty close to a ReLU (max(0, x)) zoomed out
![[notes/attachments/Pasted image 20260314014319.png]]
but with some differences around zero.

![[notes/attachments/Pasted image 20260312183337.png]]

This is the most widely used form of the FFN in high performing contemporary LLMs. I also used MoE in some instances for purposes of efficiency, routing based on the opcode to avoid redundant computation of values that are not used for the given opcode.

## The Attention Layer

```python
def softmax1(x, dim=-1):
    """Softmax with +1 in denominator - graceful missing value handling."""
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    return exp_x / (1.0 + exp_x.sum(dim=dim, keepdim=True))

def attention(Q, K, V, alibi_bias, mask):
    scores = Q @ K.T / sqrt(d) + alibi_bias + mask
    weights = softmax1(scores)  # exp(x) / (1 + sum(exp(x)))
    return weights @ V
```

We are using softmax1 and alibi, some compelling reasons for why will follow. Otherwise just plain old normal attention.

## The Building Blocks

Step functions $$SiLU(SCALE*(x+\epsilon))−SiLU(SCALE*(x−\epsilon))$$ yields a step function which can be used for comparisons
![[notes/attachments/Pasted image 20260314021120.png]]


Point indicator functions $$ f_{\varepsilon}(x) =
\frac{
\mathrm{silu}\!\left(S(x+\varepsilon)\right)
- 2\,\mathrm{silu}\!\left(Sx\right)
+ \mathrm{silu}\!\left(S(x-\varepsilon)\right)
}{
S\varepsilon\left(2\sigma(S\varepsilon)-1\right)
} $$
![[notes/attachments/Pasted image 20260314033351.png]]Which can be used for zero checking and by extension for checking other values matching, e.g. equality testing.

For lookup tables any key that we can create an indicator for we can create a one-hot vector for in the residual stream. We can then use that to select a hidden layer node, that hidden layer node can be hooked up to an value we want to populate the residual stream with.

Cancelling residuals, residuals can be canceled by having a node with a fixed gate of one using a bias of $b \approx 1.27846$ such that $SiLU(b) \approx 1$ and having the up weight from the input be 1 and the down weight be -1.

## Range checks

A range check for $x \in [a, b]$ falls out naturally from composing two step functions. Activate a step at $a$, deactivate at $b$, and subtract:

$$ R_{a,b}(x) = H_\varepsilon(x - a) - H_\varepsilon(x - b) $$

where $H_\varepsilon(x) = \operatorname{silu}(S(x+\varepsilon)) - \operatorname{silu}(S(x-\varepsilon))$ is the step function from above. Expanding:

$$ R_{a,b}(x) = \bigl[\operatorname{silu}(S(x - a + \varepsilon)) - \operatorname{silu}(S(x - a - \varepsilon))\bigr] - \bigl[\operatorname{silu}(S(x - b + \varepsilon)) - \operatorname{silu}(S(x - b - \varepsilon))\bigr] $$

This gives ~1 inside the range and ~0 outside, with transition regions of width ~$2\varepsilon$ at each boundary.

Each step function $H_\varepsilon(x - c)$ is a single SwiGLU node where the input weight shifts $x$ by $-c$ (done via bias on the gate/up projections). The range check is then just the difference of two such nodes in the down projection — one with weight +1, one with weight −1. So the entire range check is two hidden units with opposing down-projection signs.

This composes with the existing indicator function machinery: if you need "is $x$ in \[a,b\]$ AND equal to some specific value $v$", multiply the range output by the point indicator. Since both are already in $[0,1]$, the SwiGLU gate mechanism handles the multiplication natively.

With finite $S$, the transition width at each boundary is ~$2\varepsilon$, so the effective range is slightly wider than $[a,b]$. For integer-valued inputs with $\varepsilon = 0.3$ and $S \geq 10$, the error at half-integer points is negligible.

TODO cleanup and add graph
## ReLU

ReLU or rectified linear unit is just max(0, x) and can be approximated by scaling away from the non-linearities of SiLU differs from ReLU around zero, this can be used e.g. to compute an exponential but for most purposes we want it to behave like a ReLU and we can accomplish that by scaling up the inputs to get them moved away from zero. Addition and subtraction are pretty trivial by using two SiLU nodes with one negative and one positive weight of SCALE for each of the things we desire to add, fixing the gate to $\frac{1}{SCALE}$ via a bias and zero input weights. Then subtracting the negated node and adding the un-negated node for each.

## Efficient Floor

We can use the efficient range checks to get discrete values for nibbles, so we can then do it per nibble and if we need to do larger ranges we could of course do larger tables but we can also just sequentially determine the higher order bits via rescaling, subtract them out and iteratively extract the bits one nibble (or other sized chunk) at a time.

TODO make cleaner

so what we can also do to do floor on a limited range like a nibble is to enumerate the range checks for each value to that value plus one and create discrete outputs based off of that.

The MAGIC floor trick exploits floating-point representation to compute floor(x) without explicit floor operations. In IEEE 754, a float's mantissa encodes digits after the implicit leading 1. At a sufficiently large scale, the unit in last place (ULP) becomes exactly 1.0, meaning the float can only represent integers. For fp32 (23-bit mantissa), this happens at MAGIC = 2^23 = 8388608. When you compute (x + MAGIC) - MAGIC, the addition forces x into the integer-only range where it gets rounded to the nearest integer. The subsequent subtraction recovers the rounded value. With round-to-nearest-even (the default), this gives round(x), not floor(x). To get true floor for positive values, you'd subtract 0.5 first: ((x - 0.5) + MAGIC) - MAGIC. However, edge cases (exact 0.5 values) require additional care.


## Bit Range Extraction

We can extract ranges of bits from scalars by multiplying by negative powers of two, doing a floor, then doing a mod by a power of 2 corresponding to the length of the range.
## Efficient Exp

Have a key of $\sqrt{d}$ and value $e^B$ with 0 alibi slope associated with the BOS token. If we have a value N in the network we can scale it down by some SCALE and set the key to $N - B$ where B is a bias to keep the exponential in the denominator small compared to 1, we then attend to the value yielding
$$Softmax1(\frac{Q\cdot K^T}{\sqrt{d}}) \cdot V = \frac{e^{\frac{(N - B)\cdot \sqrt{d} }{\sqrt{d}}}}{1 + e^{\frac{(N - B)\cdot \sqrt{d} }{\sqrt{d}}}} \cdot e^B = \frac{e^{(N - B)}}{1 + e^{(N - B)}}  \cdot e^B  \approx e^{(N - B)} \cdot e^B = e^{N}$$

## Mixture of Experts Routing

Each opcode only needs a small subset of the FFN ops to complete so it very naturally lends itself to using Mixture of Experts (MoE) routing to avoid unnecessarily computation and masking (to avoid interference). This leads to a bit of redundancy in weights that could otherwise theoretically be shared between experts but also helps keep a clean logical seperation of functionality for each opcode. This also means that we can reuse the same dimensions of the residual streams between different opcodes/experts.
## Internal Representation

The bytecode ops load up to two values at once, all values are handled as 32 bit integers. We represent each as 16 4 bit nibbles and operate on these. This makes lookup tables for binary ops relatively small and while still allowing operations that can utilize the arithmetic internal to the transformer to take some advantage of it without too much recombination. Registers are also all loaded in the same manner in different dimensions.

## Where the Actual Computation happens
### Comparisons

The c4 opcodes $EQ$, $LT$, $GT$, $BZ$, and $BNZ$ can all be reduced to a single primitive: detecting whether a difference is zero. Let the detector input be $d$, e.g. $d = a - b$ for $EQ(a, b)$, and $d = x$ for $BZ(x)$ or $BNZ(x)$.

We implement this zero detector with three SiLU nodes sharing a common scale factor $SCALE$: Node 1 has input weight $SCALE$, bias $+SCALE \epsilon$, and output weight $+1/k$; Node 2 has input weight $SCALE$, bias $0$, and output weight $-2/k$; Node 3 has input weight $SCALE$, bias $-SCALE \epsilon$, and output weight $+1/k$, where
$$
k = SCALE \cdot \epsilon \left(2 \cdot \sigma(SCALE \cdot \epsilon) - 1\right).
$$
Thus the detector computes
$$
Z(d)=\frac{\mathrm{SiLU}(SCALE \cdot  d + SCALE \cdot  \epsilon) - 2\,\mathrm{SiLU}(SCALE \cdot d) + \mathrm{SiLU}(SCALE \cdot  d - SCALE \cdot  \epsilon)}{k}
$$
The $+1, -2, +1$ pattern is a finite second difference. For values of $d$ far from zero, all three SiLU nodes are in their approximately linear regime, so the terms cancel and $Z(d) \approx 0$. Near zero, the nonlinearity breaks this cancellation and produces a sharp bump with $Z(0) = 1$. So $Z(d)$ acts as an approximate indicator for $d = 0$.

This directly gives: $EQ(a, b)$ by applying the detector to $a - b$; $BZ(x)$ by branching when $Z(x)$ is high; and $BNZ(x)$ by branching when $Z(x)$ is low. The remaining comparisons are built from the sign of $a - b$ together with $EQ(a, b)$: $LT(a, b)$ tests whether $a - b < 0$, $GT(a, b)$ tests whether $a - b > 0$, $LE(a, b) = LT(a, b) \lor EQ(a, b)$, and $GE(a, b) = GT(a, b) \lor EQ(a, b)$.

This construction uses 11 parameters total: 9 weights and 2 biases. Rather than applying a separate zero detector to each nibble and combining the results afterward, multiple nibbles can be packed directly into the same detector by scaling the more significant nibble(s) and adding them into the node inputs. This allows biases and output weights to be shared, reducing the total parameter count. Here this is done four nibbles at a time. With $fp64$, it should in principle be possible to combine all nibbles at once.
### Basic Arithmetic

Addition, Subtraction, multiplication, left and right shift are all quite straightforward. Addition of positive numbers can be performed having input weights of SCALE for each of the numbers being added, a bias of one and zero weights for up, a weight of $\frac{1}{SCALE}$ in down then you get $$\frac{SiLU(SCALE\cdot (a+b)) \cdot  1}{SCALE} \approx a + b$$for positive a and b, subtraction naturally can work similarly. This gives addition and subtraction in 4 weights (or more formally 3 weights and 1 bias). Multiplication similarly can be done with two nodes, one with positive SCALE weight for a in gate and another with -SCALE weight in gate and both with weight 1 for b, weight $\frac{1}{SCALE}$ for down yielding $$\frac{(SiLU(SCALE\cdot a)+SiLU(-SCALE\cdot a))\cdot b}{SCALE} \approx a\cdot b$$ which accomplishes multiplication in 6 weights.

Overflow needs to be handled, which can be done by doing the arithmetic without handling it, i.e. slightly extending the bitwidth of addition, subtraction, doubling for multiplication then truncating it or ignoring it in attention, this is a bit inelegant though. Since we are doing these on the nibbles the a and b are in fact always positive. We also need to consider that using fp32 we don't have 32 bits of mantissa precision to work with. We can of course use doubles but if we want to extend to 64 bit support in the future we are really just differing the problem and I am feeling more concerned with generality here than minimality. This and robustness to quantization favor doing computation on smaller increments like nibbles or bytes. Though I admit I have been lax about prioritizing robustness to quantization.

## Shifts

Left and right shifts could simply be handled by 32 different multiplications by powers of two each, but it does not properly handle overflow, however overflow can be handled via the modulus by floor mentioned above. 
## Addition Implementation

```text
Position:     0       1       2       3
BYTE_A:      [A0]    [A1]    [A2]    [A3]
BYTE_B:      [B0]    [B1]    [B2]    [B3]
             -----   -----   -----   -----
RAW_SUM:     A0+B0   A1+B1   A2+B2   A3+B3
             (0-510 each)
RESULT:      mod256  mod256  mod256  mod256
             (0-255 each)
CARRY_OUT:   ≥256?   ≥256?   ≥256?   ≥256?
             (0 or 1)
```

Simple sweet, carrying requires a few more layers but pretty straightforward. 
## Chars

Chars can for most purposes be treated as the 8 most significant bits of an int. The exceptions being storing, loading, casting and right shifts.

## Multiplication Implementation

32-bit multiplication uses schoolbook algorithm on 4 nibbles byte-level (8-bit) reduces partial products from 36 with nibbles to 10

```
Position layout for A × B:
  A = [a0, a1, a2, a3]  (4 bytes, LSB first)
  B = [b0, b1, b2, b3]

Partial products:
  Position 0: a0*b0
  Position 1: a0*b1 + a1*b0
  Position 2: a0*b2 + a1*b1 + a2*b0
  Position 3: a0*b3 + a1*b2 + a2*b1 + a3*b0

Total: 4 + 3 + 2 + 1 = 10 products (3.6× fewer than nibble)

Each byte product: max 255 × 255 = 65025 (needs 16-bit result)
```

Total carry iterations needed for 32-bit multiply: 7 rounds (nibble) or 3 rounds (byte). For 32-bit results, products where position i+j ≥ 8 (nibble) or i+j ≥ 4 (byte) overflow beyond 32 bits and are not computed. These "guaranteed overflow" products (i+j >= threshold) would only affect bits 32-63 which are discarded in 32-bit arithmetic. Skipping them saves compute without affecting correctness.
### Division Implementation

#### Via Log

My initial idea was to do division via log and attention specifically if we can get one position to have score zero, value zero -- not needed with softmax1, and another to have score -log(n) and value 1 (and all others very low score) then the accumulated value will be $\frac{1}{n}$ the trouble is it begs the question how do you get log(n) and I ended up without a more compelling answer than Long Division, which if we are doing for the log we may as well just do directly for the division.
#### Long Division Implementation

Division is implemented as base-16 long division, each quotient digit $q \in \{0,\dots,15\}$ is computed with threshold counting $q = \sum_{k=1}^{15} \mathrm{step}(\mathrm{remainder} - k \cdot \mathrm{divisor})$. Each step contributes 1 iff the divisor fits one more time, so the sum directly gives the nibble-sized quotient digit. Operationally, each iteration does three things: update the running remainder with the next input nibble, compute the current quotient digit by counting thresholds, and subtract that digit times the divisor while writing the digit to the output. The digit computation and subtraction share the same threshold structure, so parts of the circuit can be merged for efficiency.

The quotient-digit step uses a reduced scale to avoid float32 precision problems. The step units evaluate terms like $\mathrm{silu}(S(\mathrm{remainder} - k \cdot \mathrm{divisor} + 1))$, and with 32-bit-sized values a larger scale can overflow or lose sharpness. 

Division can be a bear perhaps that is why processors like the MOS 6502 (Used in the Apple I and Apple II) didn't have a hardware div instruction at all, having users implement it as a subroutine if needed instead.
#### Via Attention With Log Sink

Start with the simple bitwise construction using 32 relevant tokens total. Let the 32 tokens correspond to bit positions $0,\dots,31$.  Set the keys to the 32 one-hot basis vectors
$$
[1,0,0,\dots,0],\quad [0,1,0,\dots,0],\quad \dots,\quad [0,\dots,0,1].
$$
Give all of these tokens value 1. If we want $\frac{1}{n}$ then we should use n - 1
$$
n-1=\sum_{k=1}^{31} b_k 2^k,
$$
then at token $n$ use query $[b_0 0\log 2, b_1 1\log 2,\; b_2 2\log 2,\; \dots,\; b_{31}31\log 2].$ Because the keys are one-hot, token $k$ gets score $b_k\,k\log 2.$ So if bit $k$ is active, it contributes $e^{k\log 2}=2^k,$ and the auxiliary tokens contribute
$$
\sum_{k=0}^{31} b_k 2^k = n-1.
$$
The $\operatorname{softmax1}$ contributes the final $1$, so its weight is $\frac{1}{1+(n-1)}=\frac{1}{n}.$ Just what we wanted. The important caveat is that inactive and unrelated tokens must be suppressed with a large negative bias or mask. Otherwise a score of $0$ would contribute $e^0=1$ and corrupt the denominator. The nibble version is the same idea in base $16$. Write
$$
n-1=\sum_{j=0}^{7}16^j d_j,
$$
where $d_j\in\{0,\dots,15\}$. Use 8 lookup positions, one per nibble, with masked tables
$$
L_j[0]=-\infty,\qquad L_j[d]=\log(16^j d)\quad\text{for }d=1,\dots,15.
$$
Selecting digit $d_j$ contributes $e^{L_j[d_j]}=16^j d_j,$ so the 8 nibble positions contribute
$$
\sum_{j=0}^{7}16^j d_j=n-1.
$$
Adding the $1$ from $\operatorname{softmax1}$ again gives weight $\frac{1}{n}.$ Now we need 8 tokens to get these keys set and for all other to have very low scores, we can detect being in the first 8 and which position using BOS as a sink, ALiBi and point indicators, so we simply use that to set the keys. There is the matter of not being able to properly divide in the first 8 tokens which is quite distressing, enough so that I am setting the default behavior to be using long division.
## Mod

Mod is implemented by using division as a subroutine, performing multiplication with the quotient and subtracting the result.
### Bitwise operations

Bitwise operations, AND, OR, XOR, NOT just lookup tables per nibble embedded in the FFNs, for AND, OR, XOR since they are binary operations and each nibble is 4 bits there need to be 256 entries, for NOT since it is unary just 16 suffices. Since the operations are the same per nibble and there is of course a lot of redundancy here that could be taken advantage of e.g. by having some convolutional FFNs, I avoided this to keep the network more vanilla. As is the per nibble operations/tables are each replicated 16 times. 

### Memory Allocation and Freeing

Memory allocation is implemented via a bump allocator, incrementing addresses by 4. Initially I ha so implementation for freeing but I figured it would probably be useful. In the spirit of this project I didn't want to do anything too hacky or adhoc, so this is what I came up with. Overwrite the value with zero. A classic solution, but how does this actually free memory? 

Well with softmax1 zero is something of a default value for attention, so we get ZFOD (zero fill on demand) for free! And so if you removed the entry from the KV cache it would have the same effect as overwriting with zero. But hooking the output of the transformer directly to the eviction policy? That sounds pretty hacky! So we just overwrite with zero and we have an eviction policy that recognizes that that means evicting both the old value and the zero overwriting it would have nil effect on the attention computation.

## File Operations

File operations pretty intrinsically require a tool call, however I wanted to make sure basic IO was possible with a 100% native version of the transformer so in addition to implementing these instructions via tool-calling I implemented support for stdin reading via reading user messages and stdout writing via system messages outside the thinking tag. The mechanism is described in the next section.
## Printing and Reading Input

Input is read directly from the user messages but some work is needed to replicate the VM interface, when the VM is ready for user input it emits an address then a special token, which indicates that the user input should be interpreted as writing bytes to sequential positions starting from that position. The LLM then gets the starting address, the offsets from them, adds them and treats each token like a write to that position as they are processed by the model.

Output is simply written to the output this involves ending and restarting the think tags for each characters. It would also probably be pretty straightforward to have it read the null terminated strings from memory but this is how it is done for now and the think tags and their content can get filtered out making things pretty seamless.

How to implement this is a somewhat more complicated, we need to be able to measure the distance from the start of the user input or the system output.

We use a fixed key at the BOS, we have multiple attention heads using the same fixed key with different alibi slopes, using these we can get different exponentials the values of which uniquely map to the position in the sequence

We have the start of user input or system response mark itself, overwriting previous ones using an attention head with significant alibi slope. 

Now we know for each token the absolute position, we can then write to the KV cache using that and we can elsewhere take a specific number and directly retrieve information associated with that sequence position. Note we only want user input and system output to use this mechanism so other items can be easily evicted from the KV cache.

The specific mechanism uses multiple attention heads sharing the same fixed key at BOS but with different ALiBi slopes. For a token at distance d from BOS, head k with slope m_k produces an attention bias of −m_k · d, giving a score contribution of exp(−m_k · d). With enough heads at different slopes, the tuple of exponential values (exp(−m_1 · d), exp(−m_2 · d), ..., exp(−m_K · d)) uniquely identifies position d — different positions produce different tuples. To retrieve the byte at position N, we construct a query that matches the exponential signature of distance N. 

Positions are encoded by ALiBi distance from the marker token, not by KV cache slot index — the exponential signature is a function of sequence distance. This means cache eviction of non-I/O tokens does not change the ALiBi distances between I/O tokens and their markers, since those tokens remain in the cache at their original positions. However, if the marker entry itself is evicted, the reference point for computing offsets is lost. 
TODO elaborate

So to take user input we have the start of a user input block, which writes to the KV cache its' location, then each user input token writes with a key corresponding to its' absolute value and a value corresponding to its' value to the KV cache. Then user input ends and there is another marker token. Then when control reverts to the VM we simply invoke a subroutine to find the start position, end position for the user input, read the absolute positions from that KV head and create a memory store for each of the user inputed values into an IO buffer in the corresponding memory. We know during the user input section we are in user input mode by detecting a user input start token without a later user input end token.

When we want to write output for the user we do much the same in reverse, we have a user output token that signals that we are writing output to the user and another indicator that shows where the buffer we are using is located in the program memory attention head, we read from that start position each time getting the offset, we then invoke a subroutine that reads from that buffer using the program memory attention head and writes them to another memory head using the same exponential addressing scheme starting from a specified start position to create an IO buffer in that head, we then pass that starting location to the attention of the USER response special token for the output tokens to attend to. Output tokens positions (starting with the user response special token since it starts generating the response), read the output start position, the IO buffer start and their absolute positions, their absolute position minus the start gives an offset.
### Position Offset Calculation

Without an efficient $\log$, we can recover offsets by a comparison cascade instead. The rough idea is: store position markers at exponential scales $2^0,2^1,\dots,2^{31}$, let the current position compare against all of them, identify the largest scale that still fits into the residual offset, subtract that contribution, and repeat. In effect, each stage recovers the next most significant bit, and the full binary offset is reconstructed from those extracted bits. This takes $O(\log N)$ sequential layers, but avoids division entirely.

With a comparison cascade, we recover the offset one bit at a time instead of computing a logarithm directly. Let the initial residual be $r^{(0)}=\mathrm{offset}$. At each stage we ask a threshold question, emit the next bit or nibble, subtract its contribution, and continue on the new residual.

For the bitwise version, layer $b$ tests whether the current residual contains the $2^b$ place: $\mathrm{bit}_b=\mathrm{step}(r^{(31-b)}-2^b)$, and then updates $r^{(32-b)}=r^{(31-b)}-\mathrm{bit}_b\,2^b$. Running this from $b=31$ down to $b=0$ gives $\mathrm{offset}=\sum_{b=0}^{31}\mathrm{bit}_b\,2^b$. So each layer extracts exactly one binary digit. This is simple and robust because every layer only has to make a binary decision, but it takes 32 sequential layers for a 32-bit offset.

A more efficient version extracts a nibble at a time by working in base $16$. At nibble position $j$, instead of testing a single threshold, we count how many multiples of $16^j$ fit into the current residual: $d_j=\sum_{k=1}^{15}\mathrm{step}(r^{(7-j)}-k\,16^j)$. This works because each threshold passed contributes $1$, so the total count is exactly the hexadecimal digit $d_j\in\{0,\dots,15\}$. We then subtract that digit’s contribution: $r^{(8-j)}=r^{(7-j)}-d_j\,16^j$. Running from $j=7$ down to $j=0$ gives $\mathrm{offset}=\sum_{j=0}^{7} d_j\,16^j$. So the nibble cascade reconstructs the full 32-bit offset in just 8 layers instead of 32.

This is mathematically the same digit-extraction pattern as long division. The only difference is that here the “divisor” at stage $j$ is just the place value $16^j$. Each nibble layer is answering the 16-way question: how many copies of $16^j$ are present in the current residual? That is exactly what the sum of step functions computes.

The nice implementation detail is that the digit extraction and subtraction can be merged into one layer, because they depend on the same thresholds. If $u_{j,k}=\mathrm{step}(r-k\,16^j)$, then the digit is $d_j=\sum_{k=1}^{15}u_{j,k}$, while the amount to subtract is $16^j\sum_{k=1}^{15}u_{j,k}=d_j\,16^j$. So one bank of units can write the digit out, and another bank with output weight $16^j$ can update the residual.

The main tradeoff is precision. In the bitwise cascade, an error at layer $b$ perturbs the residual by only $2^b$. In the nibble cascade, an error at layer $j$ can perturb it by as much as $15\cdot 16^j$, so mistakes in high-order digits are much more costly. Errors also propagate: once the residual is wrong, every later digit is extracted from the wrong value.

There is also a scale issue. The highest nibble compares against thresholds up to $15\cdot 16^7 \approx 4\times 10^9$. If the neural step is implemented as something like $\mathrm{silu}(S(r-\tau))$, then both the threshold $\tau$ and the product $S\tau$ have to remain numerically well behaved. Large thresholds force smaller usable $S$, which makes the step less sharp. That can blur nearby cases, especially if precision is limited.

The separation problem is actually hardest at the lowest nibble, not the highest. For $j=0$, adjacent thresholds differ by only $1$, so we need $S\gg 1$ to sharply distinguish neighboring digits. For large $j$, adjacent thresholds are spaced by $16^j$, so even a much smaller scale still cleanly separates them. So the nibble cascade is shallow, but it asks each layer to solve a harder classification problem.

Compared with parallel positional schemes, this cascade is fully sequential: each layer depends on the updated residual from the previous one. ALiBi can provide multiscale distance signals in parallel, and RoPE can make binary structure more explicit in the positional encoding, but the cascade is the most direct pure-FFN way to turn a scalar offset into explicit bits or nibbles. Bitwise extraction is deeper but more forgiving; nibblewise extraction is much shallower, but requires sharper thresholds and better numerical margin.
TODO edit this section

## RoPE Binary Distance Matching

With binary frequencies $\theta_k = 2^k$, each RoPE dimension tracks a different bit-scale of position. For position $p$, $\mathrm{enc}_k(p) = \alpha \cos(2^k p)$ where we choose $\alpha$ so that the full dot-product score is $\mathrm{SCALE}$ when all frequencies align. Comparing positions $p_1, p_2$ gives
$$\mathrm{score}(p_1,p_2)=\sum_k \mathrm{enc}_k(p_1)\mathrm{enc}_k(p_2)
= \alpha^2 \sum_k \cos(2^k p_1)\cos(2^k p_2)$$
If $p_1=p_2$, then $\mathrm{score}(p,p)=\alpha^2\sum_k\cos^2(2^k p)\approx\alpha^2\frac{d}{2}$ So to make the aligned score equal to $\mathrm{SCALE}$, choose $\alpha = \sqrt{\frac{2\,\mathrm{SCALE}}{d}}$ giving $\mathrm{enc}_k(p)=\sqrt{\frac{2\,\mathrm{SCALE}}{d}}\cos(2^k p)$ and therefore $\mathrm{score}(p,p)\approx\mathrm{SCALE}$. If $p_1\neq p_2$, at least one bit-scale misaligns, some terms decorrelate, and the score drops below $\mathrm{SCALE}$. So with $\theta_k=2^k$ and this normalization, RoPE acts like binary matching: exact matches score about $\mathrm{SCALE}$, and mismatched bits reduce the score. I'll take a moment here to note the difficulty here associated with keeping this vanilla if you were to use rope and hard-max attention you could simply locations as a given theta. With RoPe that pretty easily gets you relative addressing with just one head and absolute with three (treating BOS as a sink getting rotation relative to it with orthogonal unit queries then using the result to counter rotate your own relative position with the complex conjugate).
### Memset, Memcmp and Memcpy

Normally these are implemented as syscalls but you can of course just implement them as a loop, so that is what I did. You just need a starting point and length then just perform N sets, copies, or comparisons. To have these be in the transformer itself though requires they be part of the weights and not a required system prompt. No problem just have the network implement the subroutine and sure enough the simplest way for me to do that was to get the bytecode add it into the prompt and "Bake In" those operations to the weights of the network, so sure enough that is exactly what I did.

## Reading Arguments

The transformer also support argc and argv, in C4 normally these would just be set in the memory, but that would involve significant, non-transformer setup that is per-execution and not exactly transformer native. So instead if argc and argv are used the caller writes a system prompt in the following format:

| Field       | Size     | Format                 |
| ----------- | -------- | ---------------------- |
| `argc`      | 4 bytes  | Little-endian `uint32` |
| `argv[0]`   | Variable | Null-terminated string |
| `argv[1]`   | Variable | Null-terminated string |
| `...`       | `...`    | `...`                  |
| `argv[n-1]` | Variable | Null-terminated string |
Note that while this may appear like it is in some sort of equivalent of the stack calling convention it very much is not, these need to be read in the same manner as user input is and then saved to memory using the store instruction. So a subroutine has to be run, thankfully as with memset etc. we have already built most of a VM so we can just implement this in C and bake it in, so infact they are read exactly as user input is.

```c
int __argv_setup(int argv_base, int string_base) {
    int argc;
    int i;
    int ch;
    int str_ptr;

    argc = getchar();
    argc = argc + getchar() * 256;
    argc = argc + getchar() * 65536;
    argc = argc + getchar() * 16777216;

    str_ptr = string_base;
    i = 0;
    while (i < argc) {
        *(int *)(argv_base + i * 8) = str_ptr;
        ch = getchar();
        while (ch) {
            *(char *)str_ptr = ch;
            str_ptr = str_ptr + 1;
            ch = getchar();
        }
        *(char *)str_ptr = 0;
        str_ptr = str_ptr + 1;
        i = i + 1;
    }
    return argc;
}
```
TODO is this actually how it is currently implemented?
## Exiting

We want to run until completion and to implement the HALT opcode, naturally this just triggers to generation of the end of stream EOS token. Since in the tool use free version we write messages to the user and receive input from them the EOS token needs to be used for ending the message, therefore in this mode we need a special halt token to distinguish the end of the system message from the end of the program.
## Speculation

As one might imagine this transformer is highly amenable to speculative decoding, specifically by making a logical VM which outputs the tokens that we strongly suspect that the transformer will output and using that as a draft model. The draft model can pretty easily go about 1000X faster since the model is small and the speculation is perfect so very large blocks can be speculatively executed at once.
## KV Cache Pruning

As one might also imagine this model is also highly amenable to KV pruning, the process of evicting entries from a KV cache, a simple heuristic of V contribution for a query which is a unit vector in the direction of K is highly effective on account of all desired evictions for memory and registers having corresponding entries which are the same but are in more recent positions with less negative positional bias. Similarly KV merging is also highly effective.

Long running programs can easily exceed 99.999% pruning which naturally helps greatly with inference speed and efficiency. The model is relatively small and the sequences can be exceedingly long, tens to hundreds of millions of tokens or more.

###### What Gets Evicted

| Entry Type                       | Eviction Trigger        | Behavior                            |
| -------------------------------- | ----------------------- | ----------------------------------- |
| **Overwritten memory**           | Write to same address   | Old value evicted immediately       |
| **I/O entries**                  | Any new I/O write       | All previous I/O evicted            |
| **Register states**              | Any same register write | All previous same registers evicted |
| **Zero writes (how free works)** | Write zeros             | Entry evicted, no new entry         |
Eviction policy uses two complementary eviction mechanisms to keep the KV cache bounded during long programs. The primary mechanism examines key similarity and ALiBi's steep recency bias: when two cached keys are nearly identical (cosine similarity > 0.99), the older entry is already heavily downweighted relative to the newer one by ALiBi's positional penalty, and that ratio remains the roughly the same over time. The older entry will never win non-trivial attention weight — it's safe to evict. This naturally implements latest-write-wins for both memory and registers. Each VM step writes the full register file (PC, AX, SP, BP, etc.), and since each register's marker token produces a consistent key pattern across steps, the previous step's register entries get evicted as duplicates, leaving only the current values. 

The second mechanism comes into play when a zero is written, the value embedding is a zero vector, but the key is still determined by the address/register marker, so eviction works identically — the zero-valued entry replaces the old entry because the keys match, so the old overwritten value if there is one is evicted by the first mechanism. Then we simply observe that as long as there are no other keys that have non-trivial scores for that token, attending to a token with a zero vector value embedding has the same result as not attending (non-trivially) to any token under softmax1 attention. Eviction runs automatically every 120 tokens (~3 VM steps), keeping the cache at 1–10K tokens even for programs generating millions of tokens total (as long as the allocated memory and program size are small). Together, the cache grows logarithmically rather than linearly with program length, making arbitrarily long programs feasible with bounded memory.
## Sparse Tensors

Given that the network is over 99%+ sparse using sparse tensors is helpful. (Less obviously so than you might think if you aren't familiar with the efficiency of dense vs. COO sparse matrix ops). So I have included an implementation using sparse tensors which also makes the onnx files significantly smaller without any fancy compression. This also helps very significantly for self hosting since emulating the floating point operations is expensive (C4 only supports int and chars) and we of course do not have dense matrix multiply acceleration on the c4vm.
## Computational Efficiency

TODO comparison with other models, how many flops per flop, what the recursive emulation looks like, how does it compare to emulating old videogames

## Baking Prompts/Programs into the Transformer Weights

In order to actually meet the original goal we want to be able to have a program and turn it into a transformer itself, not just to have a transformer that runs bytecode from a C program. 

In order to do that we want to have that functionality in the weights of the network itself instead of relying on the prompt and attention to those bytecode tokens. So we want the weights themselves to have the same effect as attending to the tokens. Now given the structure of the network we have a rather nice property that we can exploit namely that as far as data values from the prompt/bytecode (excluding BOS) we are just using attention to retrieve a given item from the KV cache, that is we are not using mixing other than approximately 0, 1, furthermore upto overwrites we are achieving this simply by having a query equal to the key. However I will also show how we can simulate full attention using SwiGLU for completeness.

First the simple case key value retrieval, we want an exact match for the key so we take the binary key break it up into bytes or nibbles, perform an equality check for each byte or nibble, then in a subsequent layer we perform a logical AND over those results.  Note if multiple addresses have the same value for a given byte/nibble the first layer results can be shared, bounding the size needed. We can then use the outputs of the second layer as a one hot mask for the next layer, which can be implemented as a MoE router for efficiency. This efficiency is twofold. First since a proper one hot mask requires an N way logical AND, implemented by setting the weights such that the N way and is 1 iff all N are 1 and then performing a floor in an additional layer, however MoE allows us to get the top value eliminating the need for the additional layer. Second we can route to a layer that simply returns the value associated with the key, e.g. via biases, zero gate and up matrices and an identity down matrix. If multiple keys have the same value the outputs of layer 2 and their expert in layer 3 can also be shared. 

To use this in lieu of attention for the opcode fetching we simply have the network check in these FFNs instead or before the attention. Interestingly enough this effectively creates a read only code segment.

To simulate general attention is naturally more complicated. We of course are going to need dot products which can be simulated via the simple multiplication, summing with the appropriate allowances for bytes/nibbles if needed. Since the vectors are binary we can share nodes for each bit position across all simulated tokens, simply connecting the down matrix with their bit patterns. (Negative scales can be compensated with and output bias). Then we can use the efficient exponential mentioned above to compute the exponential of the dot product. 

Now we do not have efficient division, so we have to either be content with this sequentially adding that many layer to the network or find a better way. I have not successfully found a better way so this full attention baking implementation requires calculating the  $1 + \sum_i  e^{x_i}$ performing a real number version of the long division and then scaling everything down by the result. If operations in the network are sufficiently scale invariant, i.e. dependent only on relative values, positiveness, negativeness or zero checking than this computation could be avoided. This is a very significant performance hit since it means inserting a number of layers to do division for each emulated attention layer all of which need to be completed before the values can be used. If approximate scales can be used we can check e.g. for the smallest power of two larger than $1 + \sum_i  e^{x_i}$, keep a table of those and divide by that instead since division by pre-specified constants is efficient, for more precision we can use a smaller base for the exponents, however naturally the exponentials vary significantly in magnitude so this may not be feasible depending on the distribution of values.

There is also the matter of positional embeddings, which could for general attention be likewise computed by accounting for the terms. In the case of the attention we are using for bytecode we can simply treat them as read only.

## Model that Directly Runs C Code

One handy use of this baking functionality is to create a model that takes in C code and runs it, without any tool calling anywhere in the process. We simply bake into the model the bytecode for a c4 bytecode compiler and have it handoff execution to the bytecode once that is completed.

## Adapting to Different Precisions

Generally this has assumed the network is going to be run with fp32 precision, however we could target different precisions for the runtime. The main place where this consideration comes into play is in the chunk sizes used for the ALU computations, generally larger chunks are more efficient if the precision is available (bitwise operations excepting).

TODO write
## Tool Use Mode

The Neural VM supports two I/O modes. In tool calling mode, I/O opcodes (PRTF, OPEN, READ, CLOS) emit a TOOL_CALL token at the end of the VM step, which the external runner intercepts to perform the actual I/O — reading files, formatting output, etc. — before resuming execution. In neural I/O mode (the default), stdin and stdout are handled purely through transformer attention with no runner intervention. Output works via the think-tag protocol: all VM computation happens inside THINK_START/THINK_END tags (hidden from the user), and when a printf executes, the model exits the think block, emits a character byte token (visible to the user), then re-enters the think block and continues execution. Input bytes are injected into the token stream between USER_INPUT_START/END markers, and the VM reads them via attention — position-tracking heads locate the markers, and a nibble cascade extracts the offset to index into the input buffer.

Notably, runtime library functions like malloc, free, memset, and memcmp are not tool calls. They're compiled from C into the VM's bytecode and execute entirely neurally — malloc is just LEA/LI/ADD/SI, memset is a loop of SC instructions, and so on. The only operations that require external dispatch (or the neural I/O pathway) are true I/O syscalls that cross the boundary between computation and the outside world.

## Table of OPCode Neural Implementations

TODO write

## Self-Hosting

A classic test of a compiler's completeness is whether it can compile itself. For a traditional C compiler, this is straightforward:

```
gcc gcc.c -o gcc
```

For c4vm the question requires some interpretation. The system runs C programs and can compile C programs into transformers, but the transformer itself isn't a C program. A C compiler is usually written in C, a Go compiler in Go, a Rust compiler in Rust — but c4vm is a neural network. So we can't directly feed it to itself the way `gcc` compiles `gcc.c`.

What we _can_ do is close the loop through ONNX. We export the trained transformer to ONNX format, then write a minimal ONNX runtime in the subset of C that c4vm supports. This gives us self-hosting, though with more layers of indirection than the `gcc` case. Those layers are worth tracing carefully, because there are actually three distinct self-hosting relationships at work.

###### **1. The C runtime hosts itself**

c4vm interprets C programs. The ONNX runtime is a C program. So c4vm can run the program that runs c4vm:

```
c4vm onnx_runtime.c c4vm.onnx [input.c]
```

This is self-hosting in the most traditional sense — a C interpreter interpreting a C program that provides its own execution environment.

###### **2. The ONNX runtime hosts itself**

The ONNX runtime executes ONNX models. c4vm _is_ an ONNX model. So the ONNX runtime can load and execute the model that is currently interpreting it:

```
onnx_runtime c4vm.onnx onnx_runtime.c c4vm.onnx
```

The runtime's input is the very model whose execution makes the runtime possible.

###### **3. The transformer runs itself**

Combining both: the transformer interprets the C source of the ONNX runtime, which loads and executes a copy of the same transformer weights. The transformer itself can be viewed as a distinct layer of abstraction. The outer and inner models are identical — one is running as a C interpreter simulating the ONNX runtime that runs the other:

```
onnx_runtime c4vm.onnx onnx_runtime.c c4vm.onnx [input.c]
```

Each layer is a complete, functioning runtime for the layer above it. With gcc self-hosting, there's one layer of indirection: a compiler compiles itself, maybe with some allowance for the C runtime or the machine code. Here three interlocking pieces — the C runtime, the ONNX runtime, and the transformer weights — each participate in hosting themselves and the others.

TODO performance analysis of this

### Example programs

Now let's see what we can do, first a classic visualizing the MandelBrot set
TODO talk about it

This next one I wanted to have a bit of fun with we have already talked about running a chatbot with this, namely itself. But you don't normally chat by writing a C program and seeing the result. You want someone to talk to and who better than Eliza, the chatbot from 1966 that behaves like a therapist simply by asking non-directional questions. This way you can both have an LLM that can act as a therapist and have a guarantee it won't do or say anything too unhinged. And all without any training data or training!

## Bundling Programs

We of course want to be able to run programs as a single executable so I also implemented a bundler which takes the onnx runtime, the onnx weights and the target c program, compiles them into a single binary. We of course do not want to be wasteful so we also perform the bytecode compilation of the target c program during this step. And of course the bundler is runnable via c4/cllm.

This yields several useful command line utilities such as yes, cat, echo that can be used instead of the defaults to help integrate LLMs into your daily workflows.

TODO cli examples

These bundled with the C runtime, the onnx runtime implemented in C, the model weights, the bytecode for the target program come in at around 150-200KB or smaller than a size optimized rust program.

For self-hosting, there is also a C implementation of the bundler (`bundler/neural_bundler.c`) that includes an embedded C4 compiler and can itself be run under the C4 VM — meaning the neural VM can bundle programs that run on the neural VM. A fixed-point variant (`tools/neural_bundle_fixedpoint.py --program prog.c --output bundle.c`) avoids all floating point by using integer arithmetic with a $2^{12}$ scale factor and a SiLU lookup table, making the output compatible with the C4 compiler which has no `float` type. The fixed-point bundle accepts a `-n` runtime flag to toggle between native and neural arithmetic. Additional tools handle different input formats: `tools/bundle_onnx.py model.onnx output --compile` auto-converts from ONNX and optionally compiles in one step, `tools/bundle_executable.py` offers a modular `--runtime`/`--weights`/`--program` interface with a `--minimal` mode that strips out neural components entirely, and `bundler/bundle_c4.c` produces freestanding output with raw syscall wrappers instead of libc.

## Making a Quine

Much like with self hosting what exactly a Quine would be in this context requires a bit of interpretation.

One interpretation would be the transformer should output its' own weights but you do need more information about how to run it, in this case it is pretty well implicit just in the weights so it may be more interesting having it be a C program which has the transformer weights and onnx runtime in it and outputs itself, and you could say well that is just a C Quine, and you would be right, but it would also be an onnx Quine and a transformer Quine

This is a program that produces itself and does it via a transformer. But wouldn't it be nicer to have a transformer which produces itself and its' own runtime? So how would we do that, let's start with a program with the runtime, the weights, the baking logic. The program should run through the autoregressive model, it should have code to print the contents of the file as a string, (the classic string quine method), then it should use that string to bake into the weights of the base model the program for printing that string, add those to the weights in the program and only then print the whole thing.

TODO elaborate

## Reflections

I think this could be extended to have an optimizing compiler that is more directly executing the C code in the transformer, loosening the strictness of "vannillaness" would certainly help as well. The weights as they are could probably be optimized down a fair bit with relatively straightforward optimizations, for example the nibbles can be combined more aggressively to use fewer nodes for arithmetic. Memset can also be optimized by decomposing the length into powers of two and leaving the least significant bits of a memory store as 0 in the keys instead of +/- SCALE, native value fill on demand! (However this would only work on initialization since exact matches are prioritized). Self-hosting would run much more efficiently with native float support. The bundler could exclude weights for ops not used in the bytecode. If there is interest I could make a leaderboard of either smallest self hosted transformer or smallest to implement c4's vm, or steps/total ops to execute test programs.

The ease of implementing these operations in the structure of neural networks I think does bear (though quite indirectly) on how well/easily the networks can represent them and learn them. I think that is shown in the easy of ZFOD with softmax1, the ease of address overwriting with alibi, of multiplication with gated activations to name a few. On the other hand models should have an easier way to do division, log, passing of information to lower layers of the KV cache and to to do discretization (quantization may actually help with this). Additionally weight replication was used heavily and while it comes naturally in between tokens, it does not within the same token for different parts of the weight matrix, different experts or different layers, perhaps there should be a greater role for convolutions within the internals of transformers. I certainly also believe in more extensive and sophisticated weight tying. Most everything I did with SiLU was using it to approximate ReLU but that is largely a function of handwriting it. It seems feasible though that there are some features that treat it that way and experience errors when it deviates. I thought of a fair number of operations that would make things simpler using complex weights as well, so that could be a topic of some interest. Allowing control over the softmax temperature that is used for attention seems like it could be very helpful but I could see it leading to a lot of training instability. Bit depth can be used very helpfully to make operations more efficient, it is doubtful to me that these things could be effectively learned via gradient descent and generally the direction for making operations more efficient in transformers is to have less precision not more. It may be particular to trying to emulate the exact operations that it is being run with but it does go to show, precision can be used to effect! The network could be dramatically smaller and simpler with double precision but then it would have been less vanilla.

Having multiple tokens available to pre-populate the KV cache before the start of generation is probably helpful. I would imagine its' helpfulness would be rather roundabout, namely if there are mechanisms that can use some pre-populated cache, then the network would be getting punished for not having them in training and put "effort" into fixing that even though in inference it will invariably have some tokens of context before generating. Some operations are simple and require depth, it is hard to get around that while still taking advantage of parallelism well, but we can potentially allow those to be done without wasting compute either doing a whole bunch of other stuff that doesn't need to happen or by the network trying to do them in fewer layers. Thinking is bottlenecked/hobbled by single token prediction, limiting parallelism -- this can't do "kernel space" threads with real parallelism. Thinking is obviously pretty key to doing something like this, though it has persistently seemed like a kludge to me to implement it by just using the same tokens/mechanism as the autoregressive generation. Going through this exercise has only made this belief stronger. 

Some things are simpler with ALiBi others with RoPE, maybe we should be using both or some combination more often. You can probably get something like this to be learned, how much cajoling it would take is quite a question, I tried to not require values to be super exact but realistically I don't think this is at all a reasonably learnable structure. Additionally I suspect many structures would be sensitive to quantization, some require carefully chosen ALiBi slopes, though with multiple heads using different slopes it could probably be learned via a subset. The network also uses the same weights in multiple places, gradient decent learning that coordination I imagine would be difficult.

Well that was fun! 
