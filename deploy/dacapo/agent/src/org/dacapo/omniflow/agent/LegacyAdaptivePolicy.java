package org.dacapo.omniflow.agent;

/**
 * Source-faithful ADP port from Mertz and Nunes (2023).
 *
 * The archived implementation's request-type balancing branch is intentionally
 * preserved as written.  Because it returns the original Bernoulli decision in
 * both branches, it does not alter selection probability.
 */
final class LegacyAdaptivePolicy implements SamplingPolicy {
    private static final double Z = 1.96;
    private static final double P = 0.5;
    private static final double E = 0.05;

    private final double initialRate;
    private final long cycleLengthMillis;
    private final SeededBernoulli selector;
    private final boolean telemetryConsole;
    private final LegacyPerformanceBaseline performance = new LegacyPerformanceBaseline();
    private final LegacyFrequencyDataSet population = new LegacyFrequencyDataSet();
    private final LegacyFrequencyDataSet sample = new LegacyFrequencyDataSet();
    private final MemoryMeans memory = new MemoryMeans();

    private double samplingRate;
    private boolean performanceBaselineEnabled;
    private boolean reducedInPreviousBaseline;
    private long cycleStartNano;
    private long baselineWindowStartNano;
    private int cycleIndex;
    private String lastReadiness = "NOT_EVALUATED";

    LegacyAdaptivePolicy(AgentConfig config) {
        initialRate = config.uniformRate;
        samplingRate = config.uniformRate;
        cycleLengthMillis = config.cycleLengthMillis;
        selector = new SeededBernoulli(config.seed);
        telemetryConsole = config.telemetryConsole;
        long now = System.nanoTime();
        cycleStartNano = now;
        baselineWindowStartNano = now;
    }

    @Override
    public synchronized Decision decide(String requestType, double memoryDelta, long executionNanos, long second) {
        if (population.total() == 0L && !performanceBaselineEnabled) {
            cycleStartNano = System.nanoTime();
        }

        if (performanceBaselineEnabled) {
            performance.addBaseline(requestType, executionNanos);
            return new Decision(false, 0, samplingRate, samplingRate,
                    "ADP_BASELINE", "BASELINE", false);
        }

        population.add(requestType, (double) executionNanos);
        boolean simple = selector.select(samplingRate);

        // Exact archived logic.  The balancing condition is a no-op because
        // the method returns simple in both the true and false branches.
        boolean selected;
        if (simple && population.proportion(requestType) >= sample.proportion(requestType)) {
            selected = true;
        } else {
            selected = simple;
        }

        if (selected) {
            sample.add(requestType, (double) executionNanos);
        }
        memory.record(requestType, memoryDelta, selected);
        performance.addMonitoring(requestType, executionNanos);
        return new Decision(selected, 0, samplingRate, samplingRate,
                "ADP", "MONITORING", false);
    }

    @Override
    public synchronized RateUpdate onOperationsPerSecond(long second, long operationsPerSecond) {
        double previous = samplingRate;
        String status;

        if (performanceBaselineEnabled) {
            performanceBaselineEnabled = false;
            LegacyPerformanceBaseline.Comparison comparison = performance.compareBaseline();
            if (comparison.underThreshold) {
                reducedInPreviousBaseline = false;
                status = "ADP_BASELINE_KEEP";
            } else {
                samplingRate -= (samplingRate * comparison.factor) / 100.0;
                reducedInPreviousBaseline = true;
                status = "ADP_BASELINE_REDUCE";
            }
            performance.trackBaselinePerSecond(operationsPerSecond);
        } else {
            LegacyPerformanceBaseline.Comparison comparison = performance.compareMonitoring();
            if (comparison.underThreshold) {
                samplingRate += (samplingRate * comparison.factor) / 100.0;
                reducedInPreviousBaseline = false;
                status = "ADP_MONITORING_INCREASE";
            } else {
                status = "ADP_MONITORING_DEGRADED";
                if (reducedInPreviousBaseline) {
                    samplingRate -= 0.01;
                    status = "ADP_MONITORING_REDUCE_001";
                }
                long baselineAgeSeconds = Math.max(0L,
                        (System.nanoTime() - baselineWindowStartNano) / 1000000000L);
                if (baselineAgeSeconds > 3L) {
                    baselineWindowStartNano = System.nanoTime();
                    performanceBaselineEnabled = true;
                    status = "ADP_ENABLE_BASELINE";
                }
            }
            performance.trackMonitoringPerSecond(operationsPerSecond);
        }

        if (samplingRate > initialRate) samplingRate = initialRate;
        if (samplingRate < 0.01) samplingRate = 0.01;

        CycleSnapshot cycle = null;
        if (!performanceBaselineEnabled) {
            Readiness readiness = evaluateReadiness();
            lastReadiness = readiness.toString();
            if (readiness.ready) {
                cycle = closeCycle(second, false, readiness.toString(), true);
                status = status + "_CYCLE_READY";
            }
        } else {
            lastReadiness = "BASELINE_ACTIVE";
        }

        if (telemetryConsole) {
            AgentLog.info("ADP second=" + second
                    + " operations=" + operationsPerSecond
                    + " previous_rate=" + previous
                    + " effective_rate=" + samplingRate
                    + " baseline=" + performanceBaselineEnabled
                    + " readiness=" + lastReadiness
                    + " status=" + status);
        }

        return new RateUpdate(true, previous, samplingRate, "", status,
                performanceBaselineEnabled ? "BASELINE" : "MONITORING",
                lastReadiness, cycle);
    }

    @Override
    public synchronized CycleSnapshot finish(long second) {
        if (population.total() == 0L && sample.total() == 0L) return null;
        // The archived ADP run appends the final marker but does not add a final
        // mseMap value.  Preserve that distinction by emitting NaN selected
        // memory for the final ADP cycle while retaining dense diagnostics.
        return closeCycle(second, true, "FINAL_MARKER_NO_ADP_MSE", false);
    }

    private Readiness evaluateReadiness() {
        long elapsedMillis = Math.max(0L, (System.nanoTime() - cycleStartNano) / 1000000L);
        double decay = decayingConfidenceFactor(elapsedMillis);
        double precision = Z - (Z * decay);
        long minimum = minimumSampleSize(population.total(), precision);
        boolean minimumSize = sample.total() > minimum;
        boolean sameProportion = sameProportion(decay);
        boolean comparedMean = tTestEvaluation(decay);
        return new Readiness(minimumSize && sameProportion && comparedMean,
                elapsedMillis, decay, minimum, minimumSize, sameProportion, comparedMean);
    }

    private double decayingConfidenceFactor(long elapsedMillis) {
        if (elapsedMillis == 0L) return 0.999999999999;
        double value = Math.pow(0.1, (double) elapsedMillis / (double) cycleLengthMillis);
        return Math.floor(value * 10000.0) / 10000.0;
    }

    private long minimumSampleSize(long n, double precision) {
        if (n <= 1L) return 0L;
        long nInf = (long) ((precision * precision * P * (1.0 - P)) / (E * E));
        return nInf / (1L + ((nInf - 1L) / n));
    }

    private boolean sameProportion(double decay) {
        if (decay == 0.0) return true;
        for (String requestType : population.requestTypes()) {
            double pop = population.proportion(requestType);
            double sam = sample.proportion(requestType);
            double error = pop - (pop * decay);
            if (!((sam > pop) || (sam <= pop + error && sam >= pop - error))) {
                return false;
            }
        }
        return true;
    }

    private boolean tTestEvaluation(double decay) {
        RunningStats sampleFrequencies = sample.frequencyStatistics();
        if (sampleFrequencies.count() < 2L) return true;
        double variance = sampleFrequencies.variance();
        if (variance == 0.0 || Double.isNaN(variance)) return true;
        RunningStats populationFrequencies = population.frequencyStatistics();
        double populationMean = populationFrequencies.mean();
        if (sample.total() == population.total()) return true;
        double significance = 0.5 - (0.5 * decay);
        if (significance == 0.5) return true;
        return LegacyMath.oneSampleTTestReject(populationMean, sampleFrequencies, significance);
    }

    private CycleSnapshot closeCycle(
            long second, boolean finalCycle, String readiness, boolean includeSelectedMemory) {
        cycleIndex += 1;
        long elapsed = Math.max(0L, (System.nanoTime() - cycleStartNano) / 1000000L);
        double selectedMemory = includeSelectedMemory
                ? memory.selectedMeanAcrossTypes() : Double.NaN;
        CycleSnapshot snapshot = new CycleSnapshot(cycleIndex, second, finalCycle,
                population.total(), sample.total(),
                population.meanExecutionTime(), population.stdExecutionTime(),
                sample.meanExecutionTime(), sample.stdExecutionTime(),
                selectedMemory, memory.denseMeanAcrossTypes(),
                includeSelectedMemory ? memory.selectedTypes() : 0,
                memory.denseTypes(), elapsed, readiness);
        population.clear();
        sample.clear();
        memory.clear();
        cycleStartNano = System.nanoTime();
        return snapshot;
    }

    private static final class Readiness {
        final boolean ready;
        final long elapsedMillis;
        final double decay;
        final long minimumSampleSize;
        final boolean minimumSize;
        final boolean sameProportion;
        final boolean comparedMean;

        Readiness(boolean ready, long elapsedMillis, double decay,
                  long minimumSampleSize, boolean minimumSize,
                  boolean sameProportion, boolean comparedMean) {
            this.ready = ready;
            this.elapsedMillis = elapsedMillis;
            this.decay = decay;
            this.minimumSampleSize = minimumSampleSize;
            this.minimumSize = minimumSize;
            this.sameProportion = sameProportion;
            this.comparedMean = comparedMean;
        }

        @Override
        public String toString() {
            return "ready=" + ready
                    + ";elapsed_ms=" + elapsedMillis
                    + ";decay=" + decay
                    + ";minimum=" + minimumSampleSize
                    + ";has_minimum=" + minimumSize
                    + ";same_proportion=" + sameProportion
                    + ";ttest=" + comparedMean;
        }
    }
}
