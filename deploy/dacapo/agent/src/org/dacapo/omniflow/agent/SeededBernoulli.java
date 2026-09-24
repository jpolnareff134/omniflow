package org.dacapo.omniflow.agent;

import java.util.Random;

/** One shared seeded Bernoulli stream, matching the original global draw order. */
final class SeededBernoulli {
    private final Random random;

    SeededBernoulli(long seed) {
        random = new Random(seed);
    }

    synchronized boolean select(double probability) {
        if (probability <= 0.0) return false;
        if (probability >= 1.0) return true;
        return random.nextDouble() < probability;
    }
}
