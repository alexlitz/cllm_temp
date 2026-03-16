/* uniq-cllm: filter adjacent duplicate lines via neural VM */
int main() {
    int *prev;
    int *curr;
    int prev_len;
    int curr_len;
    int c;
    int same;
    int i;

    prev = malloc(2048);
    curr = malloc(2048);
    prev_len = 0 - 1;
    curr_len = 0;

    c = getchar();
    while (c >= 0) {
        if (c == 10) {
            /* Compare current with previous */
            same = 0;
            if (prev_len == curr_len) {
                same = 1;
                i = 0;
                while (i < curr_len) {
                    if (prev[i] != curr[i]) same = 0;
                    i = i + 1;
                }
            }
            /* Print if different */
            if (same == 0) {
                i = 0;
                while (i < curr_len) {
                    putchar(curr[i]);
                    i = i + 1;
                }
                putchar(10);
            }
            /* Copy current to previous */
            i = 0;
            while (i < curr_len) {
                prev[i] = curr[i];
                i = i + 1;
            }
            prev_len = curr_len;
            curr_len = 0;
        } else {
            if (curr_len < 256) {
                curr[curr_len] = c;
                curr_len = curr_len + 1;
            }
        }
        c = getchar();
    }
    return 0;
}
