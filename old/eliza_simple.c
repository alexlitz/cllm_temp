/*
 * Simple ELIZA test - minimal version for debugging
 */

char *buf;

int read_line() {
    int c;
    int i;
    i = 0;
    c = getchar();
    while (c != 10) {
        if (c < 0) {
            buf[i] = 0;
            return 0 - 1;
        }
        buf[i] = c;
        i = i + 1;
        c = getchar();
    }
    buf[i] = 0;
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

int print_str(char *s) {
    while (*s) {
        putchar(*s);
        s = s + 1;
    }
    return 0;
}

int main() {
    int n;
    buf = malloc(256);

    print_str("=================================\n");
    print_str("  ELIZA - Neural MoE Chatbot\n");
    print_str("=================================\n");
    print_str("Hello! I am ELIZA. How can I help?\n");
    print_str("(Type 'bye' to exit)\n\n");

    n = read_line();
    while (n >= 0) {
        print_str("ELIZA: ");
        if (contains(buf, "bye")) {
            print_str("Goodbye. It was nice talking to you.\n");
            n = 0 - 1;
        } else if (contains(buf, "hello")) {
            print_str("Hello! How are you feeling today?\n");
        } else if (contains(buf, "sad")) {
            print_str("I'm sorry to hear that. Why do you feel sad?\n");
        } else if (contains(buf, "happy")) {
            print_str("That's wonderful! What makes you happy?\n");
        } else if (contains(buf, "mother")) {
            print_str("Tell me more about your mother.\n");
        } else if (contains(buf, "father")) {
            print_str("Tell me more about your father.\n");
        } else if (contains(buf, "dream")) {
            print_str("What do your dreams mean to you?\n");
        } else if (contains(buf, "think")) {
            print_str("Why do you think that?\n");
        } else if (contains(buf, "feel")) {
            print_str("Tell me more about how you feel.\n");
        } else if (contains(buf, "computer")) {
            print_str("Does talking to a machine bother you?\n");
        } else if (contains(buf, "yes")) {
            print_str("You seem certain. Can you elaborate?\n");
        } else if (contains(buf, "no")) {
            print_str("Why not?\n");
        } else {
            print_str("Please go on.\n");
        }

        if (n >= 0) {
            n = read_line();
        }
    }

    print_str("=================================\n");
    free(buf);
    return 0;
}
