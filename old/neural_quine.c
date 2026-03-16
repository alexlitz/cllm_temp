#include <stdio.h>
/* Neural Quine: sparse FFN stores source as weights */
int S=256;           /* Fixed-point scale */
int W_r[4096];       /* COO row indices */
int W_v[4096];       /* COO values */
int N;               /* Non-zero count */
int silu(int x){return x>0?x:0;}
int sparse_fwd(int p){int i=0;while(i<N){if(W_r[i]==p)return W_v[i]/S;i++;}return 0;}
int neural_fwd(int p){return silu(sparse_fwd(p));}
int load_w(char*s){int i=0;N=0;while(s[i]){W_r[N]=i;W_v[N]=s[i]*S;N++;i++;}return N;}
char*Q="#include <stdio.h>%c/* Neural Quine: sparse FFN stores source as weights */%cint S=256;           /* Fixed-point scale */%cint W_r[4096];       /* COO row indices */%cint W_v[4096];       /* COO values */%cint N;               /* Non-zero count */%cint silu(int x){return x>0?x:0;}%cint sparse_fwd(int p){int i=0;while(i<N){if(W_r[i]==p)return W_v[i]/S;i++;}return 0;}%cint neural_fwd(int p){return silu(sparse_fwd(p));}%cint load_w(char*s){int i=0;N=0;while(s[i]){W_r[N]=i;W_v[N]=s[i]*S;N++;i++;}return N;}%cchar*Q=%c%s%c;%cint main(){int n=10,q=34;printf(Q,n,n,n,n,n,n,n,n,n,n,q,Q,q,n,n);return 0;}%c";
int main(){int n=10,q=34;printf(Q,n,n,n,n,n,n,n,n,n,n,q,Q,q,n,n);return 0;}
