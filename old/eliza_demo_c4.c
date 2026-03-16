/*
 * ELIZA Demo - Non-interactive version for C4 MoE VM
 *
 * Demonstrates the ELIZA chatbot with pre-defined inputs.
 * Works without getchar since MoE VM doesn't support stdin yet.
 */

char *input_buf;
int input_len;

/* Simple string copy */
int strcopy(char *dst, char *src) {
    int i;
    i = 0;
    while (src[i]) {
        dst[i] = src[i];
        i = i + 1;
    }
    dst[i] = 0;
    return i;
}

/* Simple string contains check */
int contains(char *str, char *sub) {
    int i;
    int j;
    int match;

    i = 0;
    while (str[i]) {
        j = 0;
        match = 1;
        while (sub[j]) {
            if (str[i + j] != sub[j]) {
                match = 0;
                j = 100;  /* break */
            }
            j = j + 1;
        }
        if (match) return 1;
        i = i + 1;
    }
    return 0;
}

/* Main ELIZA response logic */
int respond() {
    /* Check for goodbye */
    if (contains(input_buf, "bye")) {
        printf("ELIZA: Goodbye. Thank you for talking with me.\n");
        return 0;
    }

    /* Greetings */
    if (contains(input_buf, "hello")) {
        printf("ELIZA: Hello! How are you feeling today?\n");
        return 1;
    }

    /* Family */
    if (contains(input_buf, "mother")) {
        printf("ELIZA: Tell me more about your mother.\n");
        return 1;
    }
    if (contains(input_buf, "father")) {
        printf("ELIZA: Tell me more about your father.\n");
        return 1;
    }

    /* Feelings */
    if (contains(input_buf, "sad")) {
        printf("ELIZA: I'm sorry to hear that. Why do you feel sad?\n");
        return 1;
    }
    if (contains(input_buf, "happy")) {
        printf("ELIZA: That's wonderful! What makes you feel happy?\n");
        return 1;
    }
    if (contains(input_buf, "worry")) {
        printf("ELIZA: What worries you the most?\n");
        return 1;
    }

    /* Cognition */
    if (contains(input_buf, "think")) {
        printf("ELIZA: Why do you think that?\n");
        return 1;
    }
    if (contains(input_buf, "feel")) {
        printf("ELIZA: Tell me more about how you feel.\n");
        return 1;
    }
    if (contains(input_buf, "dream")) {
        printf("ELIZA: What do you think your dreams mean?\n");
        return 1;
    }

    /* Computer/AI */
    if (contains(input_buf, "computer")) {
        printf("ELIZA: Does it bother you that you're talking to a machine?\n");
        return 1;
    }

    /* Default */
    printf("ELIZA: Please go on.\n");
    return 1;
}

int main() {
    input_buf = malloc(256);

    printf("===========================================\n");
    printf("  ELIZA - The Classic 1966 Chatbot\n");
    printf("  Running on Neural MoE Transformer VM\n");
    printf("===========================================\n");
    printf("\n");

    /* Demo conversation */
    printf("USER: hello\n");
    strcopy(input_buf, "hello");
    respond();
    printf("\n");

    printf("USER: i feel sad today\n");
    strcopy(input_buf, "i feel sad today");
    respond();
    printf("\n");

    printf("USER: i think about my mother often\n");
    strcopy(input_buf, "i think about my mother often");
    respond();
    printf("\n");

    printf("USER: she was always worried about me\n");
    strcopy(input_buf, "she was always worried about me");
    respond();
    printf("\n");

    printf("USER: i dream about the past sometimes\n");
    strcopy(input_buf, "i dream about the past sometimes");
    respond();
    printf("\n");

    printf("USER: talking to a computer is strange\n");
    strcopy(input_buf, "talking to a computer is strange");
    respond();
    printf("\n");

    printf("USER: but it makes me happy\n");
    strcopy(input_buf, "but it makes me happy");
    respond();
    printf("\n");

    printf("USER: bye\n");
    strcopy(input_buf, "bye");
    respond();

    printf("\n");
    printf("===========================================\n");
    printf("  Session Complete\n");
    printf("===========================================\n");

    free(input_buf);
    return 0;
}
