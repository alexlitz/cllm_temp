#!/usr/bin/env python3
"""
Simple Neural Quine Generator

Creates a self-reproducing C program with embedded model.
Uses a simple encoding scheme that is easy to debug.

Usage:
    python neural_quine_gen.py model.c4onnx > quine.c
"""

import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python neural_quine_gen.py model.c4onnx > quine.c", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], 'rb') as f:
        model = list(f.read())

    ml = len(model)
    ms = ','.join(str(b) for b in model)

    # Simple quine: store the template in a data string,
    # and reconstruct it by printing the data with itself substituted

    # The DATA string uses:
    # ~ for quote
    # | for newline
    # The special marker MODEL gets replaced with model bytes
    # The special marker DATA gets replaced with the escaped DATA itself

    data = f'''/* Neural Quine */|int printf(char*f,...);int putchar(int c);char*malloc(int n);|char M[]={{MODEL}};|int L={ml};|int S=65536,*TD,*TS,*TB,NT,*NO,*NI,NN,EP;|int ri(){{int v=(M[EP]&255)|((M[EP+1]&255)<<8)|((M[EP+2]&255)<<16)|((M[EP+3]&255)<<24);EP+=4;return v;}}|int ld(){{int i,j,n,d,s,tp=0;EP=0;ri();ri();NT=ri();NN=ri();TD=malloc(500000*4);TS=malloc(256*4);TB=malloc(256*4);NO=malloc(512*4);NI=malloc(4096*4);for(i=0;i<NT;i++){{n=ri();EP+=n;d=ri();for(j=0;j<d;j++)ri();ri();s=ri();TS[i]=s;TB[i]=tp;for(j=0;j<s;j++)TD[tp++]=ri();}}for(i=0;i<NN;i++){{NO[i]=ri();n=ri();for(j=0;j<n;j++)NI[i*8+j]=ri();ri();ri();}}return 0;}}|int fm(int a,int b){{return(a/256)*(b/256);}}|int ex(int n){{int op=NO[n],*a=TD+TB[NI[n*8]],*b=TD+TB[NI[n*8+1]],*c=TD+TB[NI[n*8]],sz=TS[NI[n*8]],i;if(op==1)for(i=0;i<sz;i++)c[i]=a[i]+b[i];return 0;}}|int run(){{int i;for(i=0;i<NN;i++)ex(i);return 0;}}|int inf(int*in,int is,int*out,int os){{int i,*d=TD+TB[NT-2];for(i=0;i<is;i++)d[i]=in[i];run();d=TD+TB[NT-1];for(i=0;i<os;i++)out[i]=d[i];return 0;}}|char*D=~DATA~;|int main(){{int i,in[4],out[4];char*p;ld();in[0]=S;in[1]=2*S;in[2]=3*S;in[3]=4*S;inf(in,4,out,4);printf(~Inference: ~);for(i=0;i<4;i++)printf(~%d ~,out[i]/S);printf(~%cn~,92);p=D;while(*p){{if(*p==124)printf(~%cn~,92);else if(*p==126)putchar(34);else if(*p==77&&p[1]==79&&p[2]==68&&p[3]==69&&p[4]==76){{for(i=0;i<L;i++){{if(i)putchar(44);printf(~%d~,M[i]&255);}}p+=4;}}else if(*p==68&&p[1]==65&&p[2]==84&&p[3]==65){{char*q=D;while(*q){{if(*q==126)printf(~126~);else if(*q==124)printf(~124~);else printf(~%c~,*q);q++;}}p+=3;}}else putchar(*p);p++;}}printf(~%cn~,92);return 0;}}'''

    # Now print the actual program
    print("/* Neural Quine */")
    print("int printf(char*f,...);int putchar(int c);char*malloc(int n);")
    print(f"char M[]={{{ms}}};")
    print(f"int L={ml};")
    print("int S=65536,*TD,*TS,*TB,NT,*NO,*NI,NN,EP;")
    print("int ri(){int v=(M[EP]&255)|((M[EP+1]&255)<<8)|((M[EP+2]&255)<<16)|((M[EP+3]&255)<<24);EP+=4;return v;}")
    print("int ld(){int i,j,n,d,s,tp=0;EP=0;ri();ri();NT=ri();NN=ri();TD=malloc(500000*4);TS=malloc(256*4);TB=malloc(256*4);NO=malloc(512*4);NI=malloc(4096*4);for(i=0;i<NT;i++){n=ri();EP+=n;d=ri();for(j=0;j<d;j++)ri();ri();s=ri();TS[i]=s;TB[i]=tp;for(j=0;j<s;j++)TD[tp++]=ri();}for(i=0;i<NN;i++){NO[i]=ri();n=ri();for(j=0;j<n;j++)NI[i*8+j]=ri();ri();ri();}return 0;}")
    print("int fm(int a,int b){return(a/256)*(b/256);}")
    print("int ex(int n){int op=NO[n],*a=TD+TB[NI[n*8]],*b=TD+TB[NI[n*8+1]],*c=TD+TB[NI[n*8]],sz=TS[NI[n*8]],i;if(op==1)for(i=0;i<sz;i++)c[i]=a[i]+b[i];return 0;}")
    print("int run(){int i;for(i=0;i<NN;i++)ex(i);return 0;}")
    print("int inf(int*in,int is,int*out,int os){int i,*d=TD+TB[NT-2];for(i=0;i<is;i++)d[i]=in[i];run();d=TD+TB[NT-1];for(i=0;i<os;i++)out[i]=d[i];return 0;}")

    # Escape data for C string
    escaped_data = data.replace('"', '~')
    print(f'char*D="{escaped_data}";')

    print('int main(){int i,in[4],out[4];char*p;ld();in[0]=S;in[1]=2*S;in[2]=3*S;in[3]=4*S;inf(in,4,out,4);printf("Inference: ");for(i=0;i<4;i++)printf("%d ",out[i]/S);printf("\\n");p=D;while(*p){if(*p==124)printf("\\n");else if(*p==126)putchar(34);else if(*p==77&&p[1]==79&&p[2]==68&&p[3]==69&&p[4]==76){for(i=0;i<L;i++){if(i)putchar(44);printf("%d",M[i]&255);}p+=4;}else if(*p==68&&p[1]==65&&p[2]==84&&p[3]==65){char*q=D;while(*q){if(*q==126)printf("126");else if(*q==124)printf("124");else printf("%c",*q);q++;}p+=3;}else putchar(*p);p++;}printf("\\n");return 0;}')

if __name__ == '__main__':
    main()
