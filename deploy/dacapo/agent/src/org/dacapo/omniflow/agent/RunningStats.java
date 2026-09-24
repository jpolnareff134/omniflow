package org.dacapo.omniflow.agent;

/** Numerically stable sample statistics matching Commons Math SummaryStatistics. */
final class RunningStats {
    private long n;
    private double mean;
    private double m2;

    void add(double value) {
        n += 1L;
        double delta = value - mean;
        mean += delta / (double) n;
        double delta2 = value - mean;
        m2 += delta * delta2;
    }

    void merge(RunningStats other) {
        Snapshot snapshot = other.snapshot();
        if (snapshot.n == 0L) return;
        if (n == 0L) {
            n = snapshot.n;
            mean = snapshot.mean;
            m2 = snapshot.n < 2L ? 0.0 : snapshot.variance * (double) (snapshot.n - 1L);
            return;
        }
        long combined = n + snapshot.n;
        double delta = snapshot.mean - mean;
        double otherM2 = snapshot.n < 2L ? 0.0
                : snapshot.variance * (double) (snapshot.n - 1L);
        m2 += otherM2 + delta * delta * (double) n * (double) snapshot.n
                / (double) combined;
        mean += delta * (double) snapshot.n / (double) combined;
        n = combined;
    }

    long count() { return n; }
    double mean() { return n == 0L ? Double.NaN : mean; }
    double variance() { return n < 2L ? Double.NaN : m2 / (double) (n - 1L); }
    double std() {
        double variance = variance();
        return Double.isNaN(variance) ? Double.NaN : Math.sqrt(Math.max(0.0, variance));
    }

    void clear() {
        n = 0L;
        mean = 0.0;
        m2 = 0.0;
    }

    Snapshot snapshot() { return new Snapshot(n, mean(), variance(), std()); }

    static final class Snapshot {
        final long n;
        final double mean;
        final double variance;
        final double std;
        Snapshot(long n, double mean, double variance, double std) {
            this.n = n;
            this.mean = mean;
            this.variance = variance;
            this.std = std;
        }
    }
}
