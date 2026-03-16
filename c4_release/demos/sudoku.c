/* Sudoku Solver - C4 Compatible
 *
 * Solves Arto Inkala's "World's Hardest Sudoku" (2012) using backtracking
 * with bitmask constraints, inline constraint propagation, and MRV.
 *
 * Optimizations:
 *   1. Bitmask arrays for O(1) constraint checking
 *   2. Fused single-pass propagation + MRV with scan restart
 *   3. Unrolled candidate counting (no inner while loop)
 *   4. Inlined place/unplace to avoid function call overhead
 *   5. Dead-end detection prunes impossible states immediately
 *
 * Puzzle:
 *   8 . . | . . . | . . .
 *   . . 3 | 6 . . | . . .
 *   . 7 . | . 9 . | 2 . .
 *   ------+-------+------
 *   . 5 . | . . 7 | . . .
 *   . . . | . 4 5 | 7 . .
 *   . . . | 1 . . | . 3 .
 *   ------+-------+------
 *   . . 1 | . . . | . 6 8
 *   . . 8 | 5 . . | . 1 .
 *   . 9 . | . . . | 4 . .
 */

int *grid;
int *row_used;
int *col_used;
int *box_used;

/* Stack for undoing propagated values */
int *prop_stack;
int prop_top;

int init_grid(char *puzzle) {
    int i;
    i = 0;
    while (i < 81) {
        grid[i] = puzzle[i] - 48;
        i = i + 1;
    }
    return 0;
}

int init_bitmasks() {
    int i, row, col, num, mask, bi;
    i = 0;
    while (i < 9) {
        row_used[i] = 0;
        col_used[i] = 0;
        box_used[i] = 0;
        i = i + 1;
    }
    i = 0;
    while (i < 81) {
        num = grid[i];
        if (num > 0) {
            row = i / 9;
            col = i % 9;
            mask = 1 << num;
            bi = (row / 3) * 3 + col / 3;
            row_used[row] = row_used[row] | mask;
            col_used[col] = col_used[col] | mask;
            box_used[bi] = box_used[bi] | mask;
        }
        i = i + 1;
    }
    return 0;
}

int solve() {
    int pos, row, col, num, used, mask, bi;
    int best_pos, best_count, count, val;
    int save_top, dead;

    save_top = prop_top;
    dead = 0;
    best_pos = 81;
    best_count = 10;

    /* Single scan with restart: propagate naked singles + find MRV cell.
     * When a naked single is placed, restart from pos 0 to catch cascades. */
    pos = 0;
    while (pos < 81) {
        if (grid[pos] == 0) {
            row = pos / 9;
            col = pos % 9;
            bi = (row / 3) * 3 + col / 3;
            used = row_used[row] | col_used[col] | box_used[bi];

            /* Unrolled candidate count - avoids inner while loop overhead */
            count = 0;
            val = 0;
            if ((used & 2) == 0) { count = count + 1; val = 1; }
            if ((used & 4) == 0) { count = count + 1; val = 2; }
            if ((used & 8) == 0) { count = count + 1; val = 3; }
            if ((used & 16) == 0) { count = count + 1; val = 4; }
            if ((used & 32) == 0) { count = count + 1; val = 5; }
            if ((used & 64) == 0) { count = count + 1; val = 6; }
            if ((used & 128) == 0) { count = count + 1; val = 7; }
            if ((used & 256) == 0) { count = count + 1; val = 8; }
            if ((used & 512) == 0) { count = count + 1; val = 9; }

            if (count == 0) {
                dead = 1;
                pos = 81;
            } else if (count == 1) {
                /* Naked single: place inline and restart scan */
                mask = 1 << val;
                grid[pos] = val;
                row_used[row] = row_used[row] | mask;
                col_used[col] = col_used[col] | mask;
                box_used[bi] = box_used[bi] | mask;
                prop_stack[prop_top] = pos;
                prop_top = prop_top + 1;
                /* Restart scan to catch cascading singles */
                pos = 0 - 1;
                best_pos = 81;
                best_count = 10;
            } else if (count < best_count) {
                best_count = count;
                best_pos = pos;
            }
        }
        pos = pos + 1;
    }

    if (dead) {
        /* Undo all propagated values */
        while (prop_top > save_top) {
            prop_top = prop_top - 1;
            pos = prop_stack[prop_top];
            num = grid[pos];
            row = pos / 9;
            col = pos % 9;
            mask = 1 << num;
            bi = (row / 3) * 3 + col / 3;
            grid[pos] = 0;
            row_used[row] = row_used[row] ^ mask;
            col_used[col] = col_used[col] ^ mask;
            box_used[bi] = box_used[bi] ^ mask;
        }
        return 0;
    }

    if (best_pos >= 81) return 1;

    /* Branch on the MRV cell */
    row = best_pos / 9;
    col = best_pos % 9;
    bi = (row / 3) * 3 + col / 3;
    used = row_used[row] | col_used[col] | box_used[bi];

    num = 1;
    while (num <= 9) {
        mask = 1 << num;
        if ((used & mask) == 0) {
            /* Place inline */
            grid[best_pos] = num;
            row_used[row] = row_used[row] | mask;
            col_used[col] = col_used[col] | mask;
            box_used[bi] = box_used[bi] | mask;

            if (solve()) return 1;

            /* Unplace inline */
            grid[best_pos] = 0;
            row_used[row] = row_used[row] ^ mask;
            col_used[col] = col_used[col] ^ mask;
            box_used[bi] = box_used[bi] ^ mask;
        }
        num = num + 1;
    }

    /* Undo propagation before returning */
    while (prop_top > save_top) {
        prop_top = prop_top - 1;
        pos = prop_stack[prop_top];
        num = grid[pos];
        row = pos / 9;
        col = pos % 9;
        mask = 1 << num;
        bi = (row / 3) * 3 + col / 3;
        grid[pos] = 0;
        row_used[row] = row_used[row] ^ mask;
        col_used[col] = col_used[col] ^ mask;
        box_used[bi] = box_used[bi] ^ mask;
    }
    return 0;
}

int print_grid() {
    int r, c, val;
    r = 0;
    while (r < 9) {
        c = 0;
        while (c < 9) {
            val = grid[r * 9 + c];
            putchar(48 + val);
            if (c < 8) putchar(32);
            c = c + 1;
        }
        putchar(10);
        r = r + 1;
    }
    return 0;
}

int main() {
    int result;

    grid = malloc(81 * sizeof(int));
    row_used = malloc(9 * sizeof(int));
    col_used = malloc(9 * sizeof(int));
    box_used = malloc(9 * sizeof(int));
    prop_stack = malloc(81 * sizeof(int));
    prop_top = 0;

    init_grid("800000000003600000070090200050007000000045700000100030001000068008500010090000400");
    init_bitmasks();

    result = solve();

    if (result) {
        print_grid();
    }

    free(prop_stack);
    free(box_used);
    free(col_used);
    free(row_used);
    free(grid);
    return result;
}
