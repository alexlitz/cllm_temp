/*
 * ELIZA for MoE Neural VM - Fixed version
 * Uses printf with %s for all strings to work around MoE VM printf bug
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
        printf("%s\n", "ELIZA: Goodbye. It was nice talking to you.");
        return 0;
    }
    if (contains(input, "hello")) {
        printf("%s\n", "ELIZA: Hello! How are you feeling today?");
        return 1;
    }
    if (contains(input, "sad")) {
        printf("%s\n", "ELIZA: I'm sorry. Why do you feel sad?");
        return 1;
    }
    if (contains(input, "happy")) {
        printf("%s\n", "ELIZA: Wonderful! What makes you happy?");
        return 1;
    }
    if (contains(input, "mother")) {
        printf("%s\n", "ELIZA: Tell me about your mother.");
        return 1;
    }
    if (contains(input, "father")) {
        printf("%s\n", "ELIZA: Tell me about your father.");
        return 1;
    }
    if (contains(input, "dream")) {
        printf("%s\n", "ELIZA: What do your dreams mean?");
        return 1;
    }
    if (contains(input, "think")) {
        printf("%s\n", "ELIZA: Why do you think that?");
        return 1;
    }
    if (contains(input, "feel")) {
        printf("%s\n", "ELIZA: Tell me more about how you feel.");
        return 1;
    }
    if (contains(input, "computer")) {
        printf("%s\n", "ELIZA: Does talking to a machine bother you?");
        return 1;
    }
    printf("%s\n", "ELIZA: Please go on.");
    return 1;
}

int main() {
    input = malloc(256);

    printf("%s\n", "================================");
    printf("%s\n", "  ELIZA via MoE Neural VM");
    printf("%s\n", "================================");
    printf("%s\n", "");

    printf("%s\n", "USER: hello");
    strcopy(input, "hello");
    respond();
    printf("%s\n", "");

    printf("%s\n", "USER: i feel sad today");
    strcopy(input, "i feel sad today");
    respond();
    printf("%s\n", "");

    printf("%s\n", "USER: i think about my mother");
    strcopy(input, "i think about my mother");
    respond();
    printf("%s\n", "");

    printf("%s\n", "USER: she was always worried");
    strcopy(input, "she was always worried");
    respond();
    printf("%s\n", "");

    printf("%s\n", "USER: i had a dream about her");
    strcopy(input, "i had a dream about her");
    respond();
    printf("%s\n", "");

    printf("%s\n", "USER: talking to a computer helps");
    strcopy(input, "talking to a computer helps");
    respond();
    printf("%s\n", "");

    printf("%s\n", "USER: it makes me happy");
    strcopy(input, "it makes me happy");
    respond();
    printf("%s\n", "");

    printf("%s\n", "USER: bye");
    strcopy(input, "bye");
    respond();

    printf("%s\n", "");
    printf("%s\n", "================================");
    printf("%s\n", "  Session Complete");
    printf("%s\n", "================================");

    free(input);
    return 0;
}
