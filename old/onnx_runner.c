/*
 * ONNX Runner for C4 Transformer VM
 *
 * A minimal C program that can load and run the SwiGLU ONNX model.
 * This demonstrates that the neural multiplication can run without Python.
 *
 * Build:
 *   gcc -o onnx_runner onnx_runner.c -lm
 *
 * Usage:
 *   ./onnx_runner <a> <b>
 *
 * Example:
 *   ./onnx_runner 123 456
 *   # Output: 56088
 *
 * Note: This is a simplified implementation that computes SwiGLU directly
 * rather than loading the ONNX file. A full ONNX runtime would require
 * linking against onnxruntime-c or implementing the full ONNX parser.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * SiLU (Sigmoid Linear Unit) activation
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
double silu(double x) {
    return x / (1.0 + exp(-x));
}

/*
 * SwiGLU Multiply - exact multiplication via neural activation
 *
 * The key identity:
 *   silu(a) * b + silu(-a) * (-b) = a * b
 *
 * Proof:
 *   Let s = sigmoid(a). Then sigmoid(-a) = 1 - s.
 *   silu(a) = a * s
 *   silu(-a) = -a * (1 - s)
 *
 *   silu(a)*b + silu(-a)*(-b)
 *   = (a*s)*b + (-a*(1-s))*(-b)
 *   = a*s*b + a*(1-s)*b
 *   = a*b*(s + 1 - s)
 *   = a*b
 */
double swiglu_multiply(double a, double b) {
    return silu(a) * b + silu(-a) * (-b);
}

/*
 * Nibble addition table (4-bit + 4-bit + carry -> 4-bit + carry)
 * This would be an FFN lookup in the neural version
 */
void nibble_add(int a, int b, int carry_in, int *sum, int *carry_out) {
    int total = a + b + carry_in;
    *sum = total & 0xF;
    *carry_out = (total >> 4) & 1;
}

/*
 * 32-bit addition using nibble tables
 * This mimics the FFN-based addition in the transformer
 */
int neural_add(int a, int b) {
    int result = 0;
    int carry = 0;

    /* Process 8 nibbles (4 bits each) */
    for (int i = 0; i < 8; i++) {
        int nibble_a = (a >> (i * 4)) & 0xF;
        int nibble_b = (b >> (i * 4)) & 0xF;
        int sum, new_carry;

        nibble_add(nibble_a, nibble_b, carry, &sum, &new_carry);
        result |= (sum << (i * 4));
        carry = new_carry;
    }

    return result;
}

/*
 * Division via reciprocal table + Newton-Raphson
 * This mimics the transformer's division approach
 */
int neural_divide(int a, int b) {
    if (b == 0) return 0;

    /* Normalize divisor to [0.5, 1.0) range */
    double b_norm = (double)b;
    int shifts = 0;
    while (b_norm >= 1.0) {
        b_norm /= 2.0;
        shifts++;
    }
    while (b_norm < 0.5) {
        b_norm *= 2.0;
        shifts--;
    }

    /* Table lookup for initial reciprocal estimate (8-bit precision) */
    int table_index = (int)((b_norm - 0.5) * 512);
    if (table_index > 255) table_index = 255;
    if (table_index < 0) table_index = 0;
    double recip = 1.0 / (0.5 + table_index / 512.0);

    /* Newton-Raphson refinement: y = y * (2 - b*y) */
    for (int i = 0; i < 2; i++) {
        double by = swiglu_multiply(b_norm, recip);
        recip = swiglu_multiply(recip, 2.0 - by);
    }

    /* Scale back */
    for (int i = 0; i < shifts; i++) {
        recip /= 2.0;
    }

    /* Final multiply: a * (1/b) */
    double result = swiglu_multiply((double)a, recip);

    /* Round to integer */
    int int_result = (int)(result + 0.5);

    /* Correction loop */
    while ((int_result + 1) * b <= a) int_result++;
    while (int_result * b > a) int_result--;

    return int_result;
}

/*
 * Test the neural operations
 */
void test_ops() {
    printf("=== Neural Operations Test ===\n\n");

    /* SwiGLU multiply */
    printf("SwiGLU Multiply:\n");
    double mul_tests[][2] = {{6, 7}, {123, 456}, {999, 999}, {12345, 6789}};
    for (int i = 0; i < 4; i++) {
        double a = mul_tests[i][0];
        double b = mul_tests[i][1];
        double result = swiglu_multiply(a, b);
        int expected = (int)a * (int)b;
        printf("  %.0f * %.0f = %.0f (expected %d) %s\n",
               a, b, result, expected,
               (int)result == expected ? "PASS" : "FAIL");
    }

    printf("\nNibble Add (32-bit):\n");
    int add_tests[][2] = {{100, 200}, {255, 1}, {1000, 2000}, {123456, 654321}};
    for (int i = 0; i < 4; i++) {
        int a = add_tests[i][0];
        int b = add_tests[i][1];
        int result = neural_add(a, b);
        int expected = a + b;
        printf("  %d + %d = %d (expected %d) %s\n",
               a, b, result, expected,
               result == expected ? "PASS" : "FAIL");
    }

    printf("\nNeural Divide:\n");
    int div_tests[][2] = {{42, 7}, {100, 7}, {1000, 33}, {123456, 789}};
    for (int i = 0; i < 4; i++) {
        int a = div_tests[i][0];
        int b = div_tests[i][1];
        int result = neural_divide(a, b);
        int expected = a / b;
        printf("  %d / %d = %d (expected %d) %s\n",
               a, b, result, expected,
               result == expected ? "PASS" : "FAIL");
    }
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        /* No arguments - run tests */
        test_ops();
        return 0;
    }

    if (argc == 3) {
        /* Two arguments - compute a * b */
        double a = atof(argv[1]);
        double b = atof(argv[2]);
        double result = swiglu_multiply(a, b);
        printf("%.0f\n", result);
        return 0;
    }

    if (argc == 4) {
        /* Three arguments with operator */
        double a = atof(argv[1]);
        char op = argv[2][0];
        double b = atof(argv[3]);

        double result;
        switch (op) {
            case '*':
                result = swiglu_multiply(a, b);
                break;
            case '+':
                result = neural_add((int)a, (int)b);
                break;
            case '/':
                result = neural_divide((int)a, (int)b);
                break;
            default:
                fprintf(stderr, "Unknown operator: %c\n", op);
                return 1;
        }
        printf("%.0f\n", result);
        return 0;
    }

    fprintf(stderr, "Usage: %s [a b] or [a op b]\n", argv[0]);
    fprintf(stderr, "       %s           (run tests)\n", argv[0]);
    fprintf(stderr, "       %s 123 456   (compute 123 * 456)\n", argv[0]);
    fprintf(stderr, "       %s 100 / 7   (compute 100 / 7)\n", argv[0]);
    return 1;
}
