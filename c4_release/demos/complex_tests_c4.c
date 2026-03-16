/*
 * Complex Test Programs - C4 Compatible
 *
 * A collection of non-trivial algorithms to test the neural VM.
 * All use only c4-compatible C: no floats, while loops, if/else.
 *
 * Build with gcc:
 *   gcc -include stdio.h -include stdlib.h -o complex_tests_c4 complex_tests_c4.c
 */

/* ========== Prime Sieve (Sieve of Eratosthenes) ========== */

int *sieve;
int sieve_size;

int init_sieve(int n) {
    int i;
    sieve_size = n;
    sieve = malloc(n * 4);
    i = 0;
    while (i < n) {
        sieve[i] = 1;
        i = i + 1;
    }
    sieve[0] = 0;
    sieve[1] = 0;
    return 0;
}

int run_sieve() {
    int i;
    int j;
    int limit;

    limit = 1;
    while (limit * limit < sieve_size) {
        limit = limit + 1;
    }

    i = 2;
    while (i < limit) {
        if (sieve[i]) {
            j = i * i;
            while (j < sieve_size) {
                sieve[j] = 0;
                j = j + i;
            }
        }
        i = i + 1;
    }
    return 0;
}

int count_primes() {
    int count;
    int i;
    count = 0;
    i = 0;
    while (i < sieve_size) {
        if (sieve[i]) {
            count = count + 1;
        }
        i = i + 1;
    }
    return count;
}

int test_prime_sieve() {
    int n;
    int count;

    printf("=== Prime Sieve Test ===\n");

    n = 100;
    init_sieve(n);
    run_sieve();
    count = count_primes();
    printf("Primes up to %d: %d (expected 25)\n", n, count);

    free(sieve);

    n = 1000;
    init_sieve(n);
    run_sieve();
    count = count_primes();
    printf("Primes up to %d: %d (expected 168)\n", n, count);

    free(sieve);
    printf("\n");
    return 0;
}

/* ========== Fibonacci ========== */

int fib_memo[50];

int fib_init() {
    int i;
    i = 0;
    while (i < 50) {
        fib_memo[i] = -1;
        i = i + 1;
    }
    fib_memo[0] = 0;
    fib_memo[1] = 1;
    return 0;
}

int fib(int n) {
    if (n < 0) return 0;
    if (n > 49) return -1;
    if (fib_memo[n] >= 0) return fib_memo[n];

    fib_memo[n] = fib(n - 1) + fib(n - 2);
    return fib_memo[n];
}

int test_fibonacci() {
    int i;
    int expected[15];

    printf("=== Fibonacci Test ===\n");

    expected[0] = 0;
    expected[1] = 1;
    expected[2] = 1;
    expected[3] = 2;
    expected[4] = 3;
    expected[5] = 5;
    expected[6] = 8;
    expected[7] = 13;
    expected[8] = 21;
    expected[9] = 34;
    expected[10] = 55;
    expected[11] = 89;
    expected[12] = 144;
    expected[13] = 233;
    expected[14] = 377;

    fib_init();

    printf("Fibonacci sequence: ");
    i = 0;
    while (i < 15) {
        printf("%d ", fib(i));
        i = i + 1;
    }
    printf("\n");

    printf("fib(20) = %d (expected 6765)\n", fib(20));
    printf("fib(30) = %d (expected 832040)\n", fib(30));
    printf("\n");
    return 0;
}

/* ========== Bubble Sort ========== */

int *sort_arr;
int sort_n;

int bubble_sort() {
    int i;
    int j;
    int tmp;
    int swapped;

    i = 0;
    while (i < sort_n - 1) {
        swapped = 0;
        j = 0;
        while (j < sort_n - 1 - i) {
            if (sort_arr[j] > sort_arr[j + 1]) {
                tmp = sort_arr[j];
                sort_arr[j] = sort_arr[j + 1];
                sort_arr[j + 1] = tmp;
                swapped = 1;
            }
            j = j + 1;
        }
        if (swapped == 0) {
            return 0;
        }
        i = i + 1;
    }
    return 0;
}

int is_sorted() {
    int i;
    i = 0;
    while (i < sort_n - 1) {
        if (sort_arr[i] > sort_arr[i + 1]) {
            return 0;
        }
        i = i + 1;
    }
    return 1;
}

int test_bubble_sort() {
    int i;
    int seed;

    printf("=== Bubble Sort Test ===\n");

    sort_n = 20;
    sort_arr = malloc(sort_n * 4);

    /* Initialize with pseudo-random values using LCG */
    seed = 12345;
    i = 0;
    while (i < sort_n) {
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
        sort_arr[i] = seed % 1000;
        i = i + 1;
    }

    printf("Before: ");
    i = 0;
    while (i < sort_n) {
        printf("%d ", sort_arr[i]);
        i = i + 1;
    }
    printf("\n");

    bubble_sort();

    printf("After:  ");
    i = 0;
    while (i < sort_n) {
        printf("%d ", sort_arr[i]);
        i = i + 1;
    }
    printf("\n");

    if (is_sorted()) {
        printf("PASS: Array is sorted\n");
    } else {
        printf("FAIL: Array is NOT sorted\n");
    }

    free(sort_arr);
    printf("\n");
    return 0;
}

/* ========== Binary Search ========== */

int binary_search(int *arr, int n, int target) {
    int left;
    int right;
    int mid;

    left = 0;
    right = n - 1;

    while (left <= right) {
        mid = (left + right) / 2;
        if (arr[mid] == target) {
            return mid;
        }
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

int test_binary_search() {
    int arr[10];
    int i;
    int idx;

    printf("=== Binary Search Test ===\n");

    /* Sorted array */
    arr[0] = 2;
    arr[1] = 5;
    arr[2] = 8;
    arr[3] = 12;
    arr[4] = 16;
    arr[5] = 23;
    arr[6] = 38;
    arr[7] = 56;
    arr[8] = 72;
    arr[9] = 91;

    printf("Array: ");
    i = 0;
    while (i < 10) {
        printf("%d ", arr[i]);
        i = i + 1;
    }
    printf("\n");

    idx = binary_search(arr, 10, 23);
    printf("Search 23: index %d (expected 5)\n", idx);

    idx = binary_search(arr, 10, 2);
    printf("Search 2: index %d (expected 0)\n", idx);

    idx = binary_search(arr, 10, 91);
    printf("Search 91: index %d (expected 9)\n", idx);

    idx = binary_search(arr, 10, 50);
    printf("Search 50: index %d (expected -1)\n", idx);

    printf("\n");
    return 0;
}

/* ========== GCD (Euclidean Algorithm) ========== */

int gcd(int a, int b) {
    int tmp;
    while (b != 0) {
        tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}

int test_gcd() {
    printf("=== GCD Test ===\n");
    printf("gcd(48, 18) = %d (expected 6)\n", gcd(48, 18));
    printf("gcd(1071, 462) = %d (expected 21)\n", gcd(1071, 462));
    printf("gcd(270, 192) = %d (expected 6)\n", gcd(270, 192));
    printf("gcd(17, 13) = %d (expected 1)\n", gcd(17, 13));
    printf("gcd(100, 10) = %d (expected 10)\n", gcd(100, 10));
    printf("\n");
    return 0;
}

/* ========== Factorial ========== */

int factorial(int n) {
    int result;
    result = 1;
    while (n > 1) {
        result = result * n;
        n = n - 1;
    }
    return result;
}

int test_factorial() {
    printf("=== Factorial Test ===\n");
    printf("5! = %d (expected 120)\n", factorial(5));
    printf("7! = %d (expected 5040)\n", factorial(7));
    printf("10! = %d (expected 3628800)\n", factorial(10));
    printf("12! = %d (expected 479001600)\n", factorial(12));
    printf("\n");
    return 0;
}

/* ========== Power (Fast Exponentiation) ========== */

int power(int base, int exp) {
    int result;
    result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = result * base;
        }
        base = base * base;
        exp = exp / 2;
    }
    return result;
}

int test_power() {
    printf("=== Power Test (Fast Exponentiation) ===\n");
    printf("2^10 = %d (expected 1024)\n", power(2, 10));
    printf("3^7 = %d (expected 2187)\n", power(3, 7));
    printf("5^5 = %d (expected 3125)\n", power(5, 5));
    printf("7^4 = %d (expected 2401)\n", power(7, 4));
    printf("\n");
    return 0;
}

/* ========== Integer Square Root ========== */

int isqrt(int n) {
    int x;
    int x1;

    if (n <= 0) return 0;
    if (n == 1) return 1;

    x = n;
    x1 = (x + 1) / 2;

    while (x1 < x) {
        x = x1;
        x1 = (x + n / x) / 2;
    }

    return x;
}

int test_isqrt() {
    printf("=== Integer Square Root Test ===\n");
    printf("isqrt(16) = %d (expected 4)\n", isqrt(16));
    printf("isqrt(100) = %d (expected 10)\n", isqrt(100));
    printf("isqrt(99) = %d (expected 9)\n", isqrt(99));
    printf("isqrt(10000) = %d (expected 100)\n", isqrt(10000));
    printf("isqrt(1000000) = %d (expected 1000)\n", isqrt(1000000));
    printf("\n");
    return 0;
}

/* ========== Main ========== */

int main() {
    printf("========================================\n");
    printf("  Complex Algorithm Tests (C4)\n");
    printf("========================================\n\n");

    test_prime_sieve();
    test_fibonacci();
    test_bubble_sort();
    test_binary_search();
    test_gcd();
    test_factorial();
    test_power();
    test_isqrt();

    printf("========================================\n");
    printf("  All tests completed!\n");
    printf("========================================\n");

    return 0;
}
