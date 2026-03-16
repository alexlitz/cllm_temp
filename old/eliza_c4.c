/*
 * ELIZA - Simple chatbot for C4 MoE VM
 *
 * A minimal implementation of the classic 1966 ELIZA chatbot
 * that works within c4's limited C subset (no structs, no array initializers).
 */

char *input_buf;
int input_len;

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

/* Convert to lowercase */
int tolower_char(int c) {
    if (c >= 65) {
        if (c <= 90) {
            return c + 32;
        }
    }
    return c;
}

/* Read a line of input */
int read_line() {
    int c;
    int i;

    i = 0;
    c = getchar();
    while (c != 10) {
        if (c < 0) {
            input_buf[i] = 0;
            return 0 - 1;  /* EOF */
        }
        input_buf[i] = tolower_char(c);
        i = i + 1;
        if (i >= 255) {
            i = 254;
        }
        c = getchar();
    }
    input_buf[i] = 0;
    input_len = i;
    return i;
}

/* Main ELIZA responses */
int respond() {
    /* Check for goodbye */
    if (contains(input_buf, "bye")) {
        printf("Goodbye. Thank you for talking with me.\n");
        return 0;
    }
    if (contains(input_buf, "quit")) {
        printf("Goodbye. I hope I was helpful.\n");
        return 0;
    }

    /* Greetings */
    if (contains(input_buf, "hello")) {
        printf("Hello! How are you feeling today?\n");
        return 1;
    }
    if (contains(input_buf, "hi")) {
        printf("Hi there. What brings you here today?\n");
        return 1;
    }

    /* Family */
    if (contains(input_buf, "mother")) {
        printf("Tell me more about your mother.\n");
        return 1;
    }
    if (contains(input_buf, "father")) {
        printf("Tell me more about your father.\n");
        return 1;
    }
    if (contains(input_buf, "family")) {
        printf("How does that make you feel about your family?\n");
        return 1;
    }
    if (contains(input_buf, "sister")) {
        printf("What is your relationship with your sister like?\n");
        return 1;
    }
    if (contains(input_buf, "brother")) {
        printf("Tell me about your relationship with your brother.\n");
        return 1;
    }

    /* Feelings */
    if (contains(input_buf, "sad")) {
        printf("I'm sorry to hear that. Why do you feel sad?\n");
        return 1;
    }
    if (contains(input_buf, "happy")) {
        printf("That's wonderful! What makes you feel happy?\n");
        return 1;
    }
    if (contains(input_buf, "angry")) {
        printf("What is making you feel angry?\n");
        return 1;
    }
    if (contains(input_buf, "afraid")) {
        printf("What are you afraid of?\n");
        return 1;
    }
    if (contains(input_buf, "worry")) {
        printf("What worries you the most?\n");
        return 1;
    }
    if (contains(input_buf, "anxious")) {
        printf("Tell me more about your anxiety.\n");
        return 1;
    }
    if (contains(input_buf, "depress")) {
        printf("I understand that can be difficult. When did you start feeling this way?\n");
        return 1;
    }

    /* Cognition */
    if (contains(input_buf, "think")) {
        printf("Why do you think that?\n");
        return 1;
    }
    if (contains(input_buf, "feel")) {
        printf("Tell me more about how you feel.\n");
        return 1;
    }
    if (contains(input_buf, "believe")) {
        printf("Why do you believe that?\n");
        return 1;
    }
    if (contains(input_buf, "dream")) {
        printf("What do you think your dreams mean?\n");
        return 1;
    }
    if (contains(input_buf, "remember")) {
        printf("What else do you remember about that?\n");
        return 1;
    }

    /* Desires */
    if (contains(input_buf, "want")) {
        printf("What would it mean to you if you got what you wanted?\n");
        return 1;
    }
    if (contains(input_buf, "need")) {
        printf("Why do you feel you need that?\n");
        return 1;
    }
    if (contains(input_buf, "wish")) {
        printf("If that wish came true, how would things be different?\n");
        return 1;
    }

    /* Questions */
    if (contains(input_buf, "why")) {
        printf("Why do you ask?\n");
        return 1;
    }
    if (contains(input_buf, "how")) {
        printf("How do you think?\n");
        return 1;
    }
    if (contains(input_buf, "what")) {
        printf("What do you think?\n");
        return 1;
    }

    /* Identity */
    if (contains(input_buf, "i am")) {
        printf("How long have you been that way?\n");
        return 1;
    }
    if (contains(input_buf, "i'm")) {
        printf("How does being that way make you feel?\n");
        return 1;
    }
    if (contains(input_buf, "my")) {
        printf("Tell me more about that.\n");
        return 1;
    }

    /* Yes/No */
    if (contains(input_buf, "yes")) {
        printf("You seem certain. Can you elaborate?\n");
        return 1;
    }
    if (contains(input_buf, "no")) {
        printf("Why not?\n");
        return 1;
    }

    /* Computer/AI */
    if (contains(input_buf, "computer")) {
        printf("Does it bother you that you're talking to a machine?\n");
        return 1;
    }
    if (contains(input_buf, "machine")) {
        printf("What do you think about machines?\n");
        return 1;
    }
    if (contains(input_buf, "robot")) {
        printf("How do you feel about artificial intelligence?\n");
        return 1;
    }

    /* Problem/Help */
    if (contains(input_buf, "problem")) {
        printf("What do you think is causing this problem?\n");
        return 1;
    }
    if (contains(input_buf, "help")) {
        printf("How can I help you?\n");
        return 1;
    }

    /* Default responses - cycle through a few */
    if (input_len > 0) {
        printf("Please go on.\n");
    }
    return 1;
}

int main() {
    int running;

    input_buf = malloc(256);

    printf("ELIZA: Hello, I am ELIZA.\n");
    printf("ELIZA: How can I help you today?\n");
    printf("ELIZA: (Type 'bye' to exit)\n");
    printf("\n");

    running = 1;
    while (running) {
        printf("You: ");
        if (read_line() < 0) {
            running = 0;
        } else {
            printf("ELIZA: ");
            running = respond();
        }
    }

    free(input_buf);
    return 0;
}
