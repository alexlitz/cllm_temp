/*
 * ELIZA for MoE Neural VM
 * Uses printf (PRTF) and pre-baked conversation
 * since MoE VM doesn't support getchar
 */

char *input;

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

int contains(char *str, char *sub) {
    int i;
    int j;
    int ok;
    i = 0;
    while (str[i]) {
        j = 0;
        ok = 1;
        while (sub[j]) {
            if (str[i+j] != sub[j]) {
                ok = 0;
            }
            j = j + 1;
        }
        if (ok) return 1;
        i = i + 1;
    }
    return 0;
}

int respond() {
    if (contains(input, "bye")) {
        printf("ELIZA: Goodbye. It was nice talking to you.\n");
        return 0;
    }
    if (contains(input, "hello")) {
        printf("ELIZA: Hello! How are you feeling today?\n");
        return 1;
    }
    if (contains(input, "sad")) {
        printf("ELIZA: I'm sorry. Why do you feel sad?\n");
        return 1;
    }
    if (contains(input, "happy")) {
        printf("ELIZA: Wonderful! What makes you happy?\n");
        return 1;
    }
    if (contains(input, "mother")) {
        printf("ELIZA: Tell me about your mother.\n");
        return 1;
    }
    if (contains(input, "father")) {
        printf("ELIZA: Tell me about your father.\n");
        return 1;
    }
    if (contains(input, "dream")) {
        printf("ELIZA: What do your dreams mean?\n");
        return 1;
    }
    if (contains(input, "think")) {
        printf("ELIZA: Why do you think that?\n");
        return 1;
    }
    if (contains(input, "feel")) {
        printf("ELIZA: Tell me more about how you feel.\n");
        return 1;
    }
    if (contains(input, "computer")) {
        printf("ELIZA: Does talking to a machine bother you?\n");
        return 1;
    }
    printf("ELIZA: Please go on.\n");
    return 1;
}

int main() {
    input = malloc(256);

    printf("================================\n");
    printf("  ELIZA via MoE Neural VM\n");
    printf("================================\n");
    printf("\n");

    printf("USER: hello\n");
    strcopy(input, "hello");
    respond();
    printf("\n");

    printf("USER: i feel sad today\n");
    strcopy(input, "i feel sad today");
    respond();
    printf("\n");

    printf("USER: i think about my mother\n");
    strcopy(input, "i think about my mother");
    respond();
    printf("\n");

    printf("USER: she was always worried\n");
    strcopy(input, "she was always worried");
    respond();
    printf("\n");

    printf("USER: i had a dream about her\n");
    strcopy(input, "i had a dream about her");
    respond();
    printf("\n");

    printf("USER: talking to a computer helps\n");
    strcopy(input, "talking to a computer helps");
    respond();
    printf("\n");

    printf("USER: it makes me happy\n");
    strcopy(input, "it makes me happy");
    respond();
    printf("\n");

    printf("USER: bye\n");
    strcopy(input, "bye");
    respond();

    printf("\n");
    printf("================================\n");
    printf("  Session Complete\n");
    printf("================================\n");

    free(input);
    return 0;
}
