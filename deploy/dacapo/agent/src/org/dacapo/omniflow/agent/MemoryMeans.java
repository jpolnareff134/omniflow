package org.dacapo.omniflow.agent;

import java.util.HashMap;
import java.util.Map;

/** Positive-memory means aggregated exactly like SamplingAspect.mseMap. */
final class MemoryMeans {
    private final Map<String, PositiveStats> selected = new HashMap<String, PositiveStats>();
    private final Map<String, PositiveStats> dense = new HashMap<String, PositiveStats>();

    void record(String requestType, double memoryDelta, boolean sampled) {
        if (!(memoryDelta > 0.0)) return;
        statsFor(dense, requestType).add(memoryDelta);
        if (sampled) statsFor(selected, requestType).add(memoryDelta);
    }

    void merge(MemoryMeans other) {
        mergeMap(selected, other.selected);
        mergeMap(dense, other.dense);
    }

    double selectedMeanAcrossTypes() { return meanAcrossTypes(selected); }
    double denseMeanAcrossTypes() { return meanAcrossTypes(dense); }
    int selectedTypes() { return selected.size(); }
    int denseTypes() { return dense.size(); }

    void clear() {
        selected.clear();
        dense.clear();
    }

    private static void mergeMap(Map<String, PositiveStats> target,
                                 Map<String, PositiveStats> source) {
        for (Map.Entry<String, PositiveStats> entry : source.entrySet()) {
            PositiveStats destination = statsFor(target, entry.getKey());
            destination.count += entry.getValue().count;
            destination.sum += entry.getValue().sum;
        }
    }

    private static PositiveStats statsFor(Map<String, PositiveStats> map, String requestType) {
        PositiveStats stats = map.get(requestType);
        if (stats == null) {
            stats = new PositiveStats();
            map.put(requestType, stats);
        }
        return stats;
    }

    private static double meanAcrossTypes(Map<String, PositiveStats> map) {
        if (map.isEmpty()) return Double.NaN;
        double sum = 0.0;
        int count = 0;
        for (PositiveStats stats : map.values()) {
            if (stats.count > 0L) {
                sum += stats.sum / (double) stats.count;
                count += 1;
            }
        }
        return count == 0 ? Double.NaN : sum / (double) count;
    }

    private static final class PositiveStats {
        long count;
        double sum;
        void add(double value) { count += 1L; sum += value; }
    }
}
