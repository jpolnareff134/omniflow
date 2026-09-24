package org.dacapo.omniflow.agent;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Faithful port of PerformanceBaselineDataSet's active code paths. */
final class LegacyPerformanceBaseline {
    private static final int WINDOW = 300;

    private final ArrayDeque<Long> baselineThroughputs = new ArrayDeque<Long>();
    private final Map<Long, Map<String, RunningStats.Snapshot>> baselineByThroughput =
            new HashMap<Long, Map<String, RunningStats.Snapshot>>();
    private final Map<String, RunningStats> currentBaseline = new HashMap<String, RunningStats>();

    private final ArrayDeque<Long> monitoringThroughputs = new ArrayDeque<Long>();
    private final Map<Long, Map<String, RunningStats.Snapshot>> monitoringByThroughput =
            new HashMap<Long, Map<String, RunningStats.Snapshot>>();
    private final Map<String, RunningStats> currentMonitoring = new HashMap<String, RunningStats>();

    void addBaseline(String requestType, long executionNanos) {
        statsFor(currentBaseline, requestType).add((double) executionNanos);
    }

    void addMonitoring(String requestType, long executionNanos) {
        statsFor(currentMonitoring, requestType).add((double) executionNanos);
    }

    void trackBaselinePerSecond(long operationsPerSecond) {
        if (operationsPerSecond == 0L) return;
        addWindowValue(baselineThroughputs, operationsPerSecond);
        baselineByThroughput.put(Long.valueOf(operationsPerSecond), snapshotAndClear(currentBaseline));
    }

    void trackMonitoringPerSecond(long operationsPerSecond) {
        if (operationsPerSecond == 0L) return;
        addWindowValue(monitoringThroughputs, operationsPerSecond);
        monitoringByThroughput.put(Long.valueOf(operationsPerSecond), snapshotAndClear(currentMonitoring));
    }

    Comparison compareBaseline() {
        return compare(baselineThroughputs, baselineByThroughput, currentBaseline);
    }

    Comparison compareMonitoring() {
        return compare(monitoringThroughputs, monitoringByThroughput, currentMonitoring);
    }

    private static Comparison compare(
            ArrayDeque<Long> throughputWindow,
            Map<Long, Map<String, RunningStats.Snapshot>> byThroughput,
            Map<String, RunningStats> current) {
        if (throughputWindow.isEmpty()) return new Comparison(true, 0.0, 0L, 0L);
        Long median = medianValue(throughputWindow);
        Map<String, RunningStats.Snapshot> normal = byThroughput.get(median);
        if (normal == null || normal.isEmpty()) return new Comparison(true, 0.0, 0L, 0L);

        long failed = 0L;
        long success = 0L;
        double maxFactor = Double.NaN;
        long compared = 0L;
        for (Map.Entry<String, RunningStats.Snapshot> entry : normal.entrySet()) {
            RunningStats currentStats = current.get(entry.getKey());
            RunningStats.Snapshot normalStats = entry.getValue();
            if (currentStats == null || currentStats.count() <= 2L || normalStats.n <= 2L) continue;
            compared += 1L;
            double currentMean = currentStats.mean();
            double threshold = normalStats.mean + normalStats.std;
            if (currentMean > threshold) {
                failed += 1L;
            } else {
                double factor = (normalStats.mean - currentMean) / currentMean;
                if (Double.isNaN(maxFactor) || factor > maxFactor) maxFactor = factor;
                success += 1L;
            }
        }
        double factor = Double.isNaN(maxFactor) ? 0.0 : maxFactor;
        if (compared < 2L) return new Comparison(true, factor, success, failed);
        if (success + failed > 1000L) return new Comparison(success > failed, factor, success, failed);
        return new Comparison(failed <= 2L, factor, success, failed);
    }

    private static RunningStats statsFor(Map<String, RunningStats> map, String requestType) {
        RunningStats stats = map.get(requestType);
        if (stats == null) {
            stats = new RunningStats();
            map.put(requestType, stats);
        }
        return stats;
    }

    private static Map<String, RunningStats.Snapshot> snapshotAndClear(Map<String, RunningStats> source) {
        Map<String, RunningStats.Snapshot> result = new HashMap<String, RunningStats.Snapshot>();
        for (Map.Entry<String, RunningStats> entry : source.entrySet()) {
            result.put(entry.getKey(), entry.getValue().snapshot());
        }
        source.clear();
        return result;
    }

    private static void addWindowValue(ArrayDeque<Long> window, long value) {
        if (window.size() >= WINDOW) window.removeFirst();
        window.addLast(Long.valueOf(value));
    }

    private static Long medianValue(ArrayDeque<Long> values) {
        List<Long> sorted = new ArrayList<Long>(values);
        Collections.sort(sorted);
        return sorted.get(sorted.size() / 2);
    }

    static final class Comparison {
        final boolean underThreshold;
        final double factor;
        final long success;
        final long failed;
        Comparison(boolean underThreshold, double factor, long success, long failed) {
            this.underThreshold = underThreshold;
            this.factor = factor;
            this.success = success;
            this.failed = failed;
        }
    }
}
