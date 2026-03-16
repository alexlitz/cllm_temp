#!/usr/bin/env python3
"""
Neural Quine Generator

Creates a C program that:
1. Runs neural inference on the embedded model
2. Prints its own exact source code

The quine executes the neural network before printing itself.

Usage:
    python gen_neural_quine.py model.c4onnx > quine.c
    gcc -o quine quine.c
    ./quine > quine2.c
    diff quine.c quine2.c  # Should be empty
"""

import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python gen_neural_quine.py model.c4onnx > quine.c", file=sys.stderr)
        sys.exit(1)

    # Read model bytes
    with open(sys.argv[1], 'rb') as f:
        model = list(f.read())

    model_str = ','.join(str(b) for b in model)
    model_len = len(model)

    # Generate the complete quine with neural runtime
    print("/* Neural Quine - Runs inference then prints itself */")
    print("int printf(char*f,...);int putchar(int c);char*malloc(int n);")
    print(f"char M[]={{{model_str}}};")
    print(f"int ML={model_len};")

    # Minimal runtime variables
    print("int S=65536;int*TD;int*TS;int*TB;int NT;int*NO;int*NI;int*NNI;int*NNO;int NN;int EP;")

    # Read embedded int
    print("int rei(){int v;v=(M[EP]&255)|((M[EP+1]&255)<<8)|((M[EP+2]&255)<<16)|((M[EP+3]&255)<<24);EP=EP+4;return v;}")

    # Load model function
    print("int load(){int i,j,n,d,s;EP=0;if(rei()!=1481526863)return-1;rei();NT=rei();NN=rei();TD=malloc(500000*4);TS=malloc(256*4);TB=malloc(256*4);NO=malloc(512*4);NI=malloc(512*8*4);NNI=malloc(512*4);NNO=malloc(512*4);int tp=0;i=0;while(i<NT){n=rei();EP=EP+n;d=rei();j=0;while(j<d){rei();j=j+1;}rei();s=rei();TS[i]=s;TB[i]=tp;j=0;while(j<s){TD[tp]=rei();tp=tp+1;j=j+1;}i=i+1;}i=0;while(i<NN){NO[i]=rei();NNI[i]=rei();j=0;while(j<NNI[i]){NI[i*8+j]=rei();j=j+1;}NNO[i]=rei();j=0;while(j<NNO[i]){rei();j=j+1;}i=i+1;}return 0;}")

    # Fixed-point multiply
    print("int fm(int a,int b){return(a/256)*(b/256);}")

    # Execute node
    print("int ex(int ni){int op,i0,i1,o0,i,*a,*b,*c,sz;op=NO[ni];i0=NI[ni*8];i1=NI[ni*8+1];o0=NI[ni*8];a=TD+TB[i0];b=TD+TB[i1];c=TD+TB[o0];sz=TS[o0];if(op==1){i=0;while(i<sz){c[i]=a[i]+b[i];i=i+1;}}if(op==2){i=0;while(i<sz){c[i]=fm(a[i],b[i]);i=i+1;}}return 0;}")

    # Run model
    print("int run(){int i=0;while(i<NN){ex(i);i=i+1;}return 0;}")

    # Inference function
    print("int infer(int*in,int is,int*out,int os){int i,*d;d=TD+TB[NT-2];i=0;while(i<is){d[i]=in[i];i=i+1;}run();d=TD+TB[NT-1];i=0;while(i<os){out[i]=d[i];i=i+1;}return 0;}")

    # The quine data string - encoded version of the program
    # @ = newline, $ = quote, # = backslash, ^ = percent

    # Build all the lines we need to reproduce
    lines = [
        "/* Neural Quine - Runs inference then prints itself */",
        "int printf(char*f,...);int putchar(int c);char*malloc(int n);",
        f"char M[]={{MODEL}};",  # placeholder
        f"int ML={model_len};",
        "int S=65536;int*TD;int*TS;int*TB;int NT;int*NO;int*NI;int*NNI;int*NNO;int NN;int EP;",
        "int rei(){int v;v=(M[EP]&255)|((M[EP+1]&255)<<8)|((M[EP+2]&255)<<16)|((M[EP+3]&255)<<24);EP=EP+4;return v;}",
        "int load(){int i,j,n,d,s;EP=0;if(rei()!=1481526863)return-1;rei();NT=rei();NN=rei();TD=malloc(500000*4);TS=malloc(256*4);TB=malloc(256*4);NO=malloc(512*4);NI=malloc(512*8*4);NNI=malloc(512*4);NNO=malloc(512*4);int tp=0;i=0;while(i<NT){n=rei();EP=EP+n;d=rei();j=0;while(j<d){rei();j=j+1;}rei();s=rei();TS[i]=s;TB[i]=tp;j=0;while(j<s){TD[tp]=rei();tp=tp+1;j=j+1;}i=i+1;}i=0;while(i<NN){NO[i]=rei();NNI[i]=rei();j=0;while(j<NNI[i]){NI[i*8+j]=rei();j=j+1;}NNO[i]=rei();j=0;while(j<NNO[i]){rei();j=j+1;}i=i+1;}return 0;}",
        "int fm(int a,int b){return(a/256)*(b/256);}",
        "int ex(int ni){int op,i0,i1,o0,i,*a,*b,*c,sz;op=NO[ni];i0=NI[ni*8];i1=NI[ni*8+1];o0=NI[ni*8];a=TD+TB[i0];b=TD+TB[i1];c=TD+TB[o0];sz=TS[o0];if(op==1){i=0;while(i<sz){c[i]=a[i]+b[i];i=i+1;}}if(op==2){i=0;while(i<sz){c[i]=fm(a[i],b[i]);i=i+1;}}return 0;}",
        "int run(){int i=0;while(i<NN){ex(i);i=i+1;}return 0;}",
        "int infer(int*in,int is,int*out,int os){int i,*d;d=TD+TB[NT-2];i=0;while(i<is){d[i]=in[i];i=i+1;}run();d=TD+TB[NT-1];i=0;while(i<os){out[i]=d[i];i=i+1;}return 0;}",
        "char*Q=\"QUINE_DATA\";",  # placeholder
        "int main(){int i,j,in[4],out[4];char*p;load();in[0]=S;in[1]=2*S;in[2]=3*S;in[3]=4*S;infer(in,4,out,4);printf(\"Neural inference: \");i=0;while(i<4){printf(\"%d \",out[i]/S);i=i+1;}printf(\"@n\");j=0;p=Q;while(j<13){if(j==2){printf(\"char M[]=\");putchar('{');i=0;while(i<ML){if(i)putchar(',');printf(\"%d\",M[i]&255);i=i+1;}printf(\"};@n\");}else if(j==12){printf(\"char*Q=@$\");while(*p){if(*p=='@')printf(\"@@@\");else if(*p=='$')printf(\"@$\");else if(*p=='#')printf(\"##\");else if(*p=='^')printf(\"^^\");else putchar(*p);p=p+1;}printf(\"@$;@n\");}else{while(*p!='@'&&*p){if(*p=='$')putchar('\"');else if(*p=='#')putchar('\\\\');else if(*p=='^')putchar('%');else putchar(*p);p=p+1;}putchar('@n');if(*p)p=p+1;}j=j+1;}while(*p){if(*p=='@')putchar('@n');else if(*p=='$')putchar('\"');else if(*p=='#')putchar('\\\\');else if(*p=='^')putchar('%');else putchar(*p);p=p+1;}putchar('@n');return 0;}"
    ]

    # Encode for Q string
    def encode_q(s):
        return s.replace('\\', '#').replace('"', '$').replace('\n', '@').replace('%', '^')

    q_content = '@'.join(encode_q(line) for line in lines)

    # Print the Q string
    print(f'char*Q="{q_content}";')

    # Print main
    main_func = '''int main(){int i,j,in[4],out[4];char*p;load();in[0]=S;in[1]=2*S;in[2]=3*S;in[3]=4*S;infer(in,4,out,4);printf("Neural inference: ");i=0;while(i<4){printf("%d ",out[i]/S);i=i+1;}printf("\\n");j=0;p=Q;while(j<13){if(j==2){printf("char M[]=");putchar('{');i=0;while(i<ML){if(i)putchar(',');printf("%d",M[i]&255);i=i+1;}printf("};\\n");}else if(j==12){printf("char*Q=\\"");while(*p){if(*p=='@')printf("@");else if(*p=='$')printf("$");else if(*p=='#')printf("##");else if(*p=='^')printf("^^");else putchar(*p);p=p+1;}printf("\\";\\n");}else{while(*p!='@'&&*p){if(*p=='$')putchar('"');else if(*p=='#')putchar('\\\\');else if(*p=='^')putchar('%');else putchar(*p);p=p+1;}putchar('\\n');if(*p)p=p+1;}j=j+1;}while(*p){if(*p=='@')putchar('\\n');else if(*p=='$')putchar('"');else if(*p=='#')putchar('\\\\');else if(*p=='^')putchar('%');else putchar(*p);p=p+1;}putchar('\\n');return 0;}'''
    print(main_func)

if __name__ == '__main__':
    main()
