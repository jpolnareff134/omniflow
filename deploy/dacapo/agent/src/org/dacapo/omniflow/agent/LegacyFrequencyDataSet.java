package org.dacapo.omniflow.agent;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/** Faithful Java-8 port of the original FrequencyDataSet. */
final class LegacyFrequencyDataSet {
    private final Map<String, Long> counts = new HashMap<String, Long>();
    private final RunningStats executionTimes = new RunningStats();
    private long total;

    void add(String requestType, double executionNanos) {
        Long old = counts.get(requestType);
        counts.put(requestType, Long.valueOf(old == null ? 1L : old.longValue() + 1L));
        executionTimes.add(executionNanos);
        total += 1L;
    }

    void merge(LegacyFrequencyDataSet other) {
        for (Map.Entry<String, Long> entry : other.counts.entrySet()) {
            Long old = counts.get(entry.getKey());
            counts.put(entry.getKey(), Long.valueOf(
                    (old == null ? 0L : old.longValue()) + entry.getValue().longValue()));
        }
        executionTimes.merge(other.executionTimes);
        total += other.total;
    }

    long total() { return total; }
    double meanExecutionTime() { return executionTimes.mean(); }
    double stdExecutionTime() { return executionTimes.std(); }
    Set<String> requestTypes() { return counts.keySet(); }

    double proportion(String requestType) {
        if (total == 0L) return 0.0;
        Long value = counts.get(requestType);
        return value == null ? 0.0 : (double) value.longValue() / (double) total;
    }

    RunningStats frequencyStatistics() {
        RunningStats statistics = new RunningStats();
        for (Long count : counts.values()) {
            statistics.add(count.doubleValue());
        }
        return statistics;
    }

    void clear() {
        counts.clear();
        executionTimes.clear();
        total = 0L;
    }
}
