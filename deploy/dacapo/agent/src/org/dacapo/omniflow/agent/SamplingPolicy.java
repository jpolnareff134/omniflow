package org.dacapo.omniflow.agent;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

/** Sampling policies used by the agent. */
public interface SamplingPolicy {
    Decision decide(String requestType, double value, long executionNanos, long second);

    default RateUpdate onOperationsPerSecond(long second, long operationsPerSecond) {
        return RateUpdate.notApplicable();
    }

    default CycleSnapshot finish(long second) {
        return null;
    }

    default List<CycleSnapshot> finishCycles(long second) {
        CycleSnapshot cycle = finish(second);
        if (cycle == null) return Collections.emptyList();
        return Collections.singletonList(cycle);
    }

    default List<ControllerSnapshot> controllerSnapshots() {
        return Collections.emptyList();
    }

    static SamplingPolicy create(AgentConfig config) {
        SamplingPolicy base;
        if ("full".equals(config.policy)) {
            base = new FullPolicy();
        } else if ("uni".equals(config.policy)) {
            base = new UniformPolicy(config.uniformRate, config.seed);
        } else if ("inv".equals(config.policy)) {
            base = new InversePolicy(config.uniformRate, config.seed, config.telemetryConsole);
        } else if ("adp".equals(config.policy)) {
            return new LegacyAdaptivePolicy(config);
        } else {
            base = new OmniFlowPolicy(config);
        }
        List<Long> markers = CycleMarkers.load(config.cycleMarkersFile);
        return markers.isEmpty() ? base : new CycleTrackingPolicy(base, markers);
    }

    final class Decision {
        public final boolean sampled;
        public final int interval;
        public final double rate;
        public final double urgency;
        public final String status;
        public final String baselineState;
        public final boolean cycleMarker;

        Decision(boolean sampled, int interval, double rate, double urgency,
                 String status, String baselineState, boolean cycleMarker) {
            this.sampled = sampled;
            this.interval = interval;
            this.rate = rate;
            this.urgency = urgency;
            this.status = status;
            this.baselineState = baselineState;
            this.cycleMarker = cycleMarker;
        }
    }

    final class RateUpdate {
        public final boolean applicable;
        public final double previousRate;
        public final double effectiveRate;
        public final String legacyComputedRate;
        public final String status;
        public final String baselineState;
        public final String readinessState;
        public final CycleSnapshot cycle;

        RateUpdate(boolean applicable, double previousRate, double effectiveRate,
                   String legacyComputedRate, String status, String baselineState,
                   String readinessState, CycleSnapshot cycle) {
            this.applicable = applicable;
            this.previousRate = previousRate;
            this.effectiveRate = effectiveRate;
            this.legacyComputedRate = legacyComputedRate;
            this.status = status;
            this.baselineState = baselineState;
            this.readinessState = readinessState;
            this.cycle = cycle;
        }

        static RateUpdate notApplicable() {
            return new RateUpdate(false, Double.NaN, Double.NaN, "", "NOT_APPLICABLE",
                    "MONITORING", "", null);
        }

        RateUpdate withCycle(CycleSnapshot snapshot) {
            return new RateUpdate(applicable, previousRate, effectiveRate,
                    legacyComputedRate, status, baselineState, readinessState, snapshot);
        }
    }

    final class CycleSnapshot {
        public final int cycleIndex;
        public final long markerSecond;
        public final boolean finalCycle;
        public final long populationSize;
        public final long sampleSize;
        public final double populationMeanNanos;
        public final double populationStdNanos;
        public final double sampleMeanNanos;
        public final double sampleStdNanos;
        public final double selectedMemoryMeanBytes;
        public final double denseMemoryMeanBytes;
        public final int selectedMemoryTypes;
        public final int denseMemoryTypes;
        public final long elapsedMillis;
        public final String readiness;

        CycleSnapshot(int cycleIndex, long markerSecond, boolean finalCycle,
                      long populationSize, long sampleSize,
                      double populationMeanNanos, double populationStdNanos,
                      double sampleMeanNanos, double sampleStdNanos,
                      double selectedMemoryMeanBytes, double denseMemoryMeanBytes,
                      int selectedMemoryTypes, int denseMemoryTypes,
                      long elapsedMillis, String readiness) {
            this.cycleIndex = cycleIndex;
            this.markerSecond = markerSecond;
            this.finalCycle = finalCycle;
            this.populationSize = populationSize;
            this.sampleSize = sampleSize;
            this.populationMeanNanos = populationMeanNanos;
            this.populationStdNanos = populationStdNanos;
            this.sampleMeanNanos = sampleMeanNanos;
            this.sampleStdNanos = sampleStdNanos;
            this.selectedMemoryMeanBytes = selectedMemoryMeanBytes;
            this.denseMemoryMeanBytes = denseMemoryMeanBytes;
            this.selectedMemoryTypes = selectedMemoryTypes;
            this.denseMemoryTypes = denseMemoryTypes;
            this.elapsedMillis = elapsedMillis;
            this.readiness = readiness;
        }
    }

    final class ControllerSnapshot {
        public final String requestType;
        public final String signalMode;
        public final long decisions;
        public final long sampled;
        public final long skipped;
        public final long initCount;
        public final long normalCount;
        public final long outlierCount;
        public final long driftResetCount;
        public final long rawHeapPositive;
        public final long rawHeapZero;
        public final long rawHeapNegative;
        public final boolean calibrated;
        public final long calibratedAtDecision;
        public final long calibratedAtSample;
        public final int minInterval;
        public final double meanInterval;
        public final int maxInterval;
        public final double meanUrgency;
        public final double finalUrgency;
        public final int finalInterval;
        public final double sigmaRef;
        public final double trackerStd;
        public final double sampledSignalMean;
        public final double sampledSignalStd;

        ControllerSnapshot(String requestType, String signalMode, long decisions, long sampled,
                           long skipped, long initCount, long normalCount, long outlierCount,
                           long driftResetCount, long rawHeapPositive, long rawHeapZero,
                           long rawHeapNegative, boolean calibrated, long calibratedAtDecision,
                           long calibratedAtSample, int minInterval, double meanInterval,
                           int maxInterval, double meanUrgency, double finalUrgency,
                           int finalInterval, double sigmaRef, double trackerStd,
                           double sampledSignalMean, double sampledSignalStd) {
            this.requestType = requestType;
            this.signalMode = signalMode;
            this.decisions = decisions;
            this.sampled = sampled;
            this.skipped = skipped;
            this.initCount = initCount;
            this.normalCount = normalCount;
            this.outlierCount = outlierCount;
            this.driftResetCount = driftResetCount;
            this.rawHeapPositive = rawHeapPositive;
            this.rawHeapZero = rawHeapZero;
            this.rawHeapNegative = rawHeapNegative;
            this.calibrated = calibrated;
            this.calibratedAtDecision = calibratedAtDecision;
            this.calibratedAtSample = calibratedAtSample;
            this.minInterval = minInterval;
            this.meanInterval = meanInterval;
            this.maxInterval = maxInterval;
            this.meanUrgency = meanUrgency;
            this.finalUrgency = finalUrgency;
            this.finalInterval = finalInterval;
            this.sigmaRef = sigmaRef;
            this.trackerStd = trackerStd;
            this.sampledSignalMean = sampledSignalMean;
            this.sampledSignalStd = sampledSignalStd;
        }
    }

    final class FullPolicy implements SamplingPolicy {
        @Override
        public Decision decide(String requestType, double value, long executionNanos, long second) {
            return new Decision(true, 1, 1.0, 1.0, "FULL", "MONITORING", false);
        }
    }

    /** Fixed-rate global Bernoulli stream, matching the original UNI shape. */
    final class UniformPolicy implements SamplingPolicy {
        private final double rate;
        private final SeededBernoulli selector;

        UniformPolicy(double rate, long seed) {
            this.rate = rate;
            this.selector = new SeededBernoulli(seed);
        }

        @Override
        public Decision decide(String requestType, double value, long executionNanos, long second) {
            boolean selected = selector.select(rate);
            return new Decision(selected, 0, rate, rate, "UNI", "MONITORING", false);
        }
    }

    /** Throughput-inverse policy from the archived Mertz--Nunes source. */
    final class InversePolicy implements SamplingPolicy {
        private final double initialRate;
        private final SeededBernoulli selector;
        private final boolean telemetryConsole;
        private volatile double currentRate;
        private double minimumThroughput = Double.POSITIVE_INFINITY;
        private double maximumThroughput = Double.NEGATIVE_INFINITY;

        InversePolicy(double initialRate, long seed, boolean telemetryConsole) {
            this.initialRate = initialRate;
            this.currentRate = initialRate;
            this.selector = new SeededBernoulli(seed);
            this.telemetryConsole = telemetryConsole;
        }

        @Override
        public Decision decide(String requestType, double value, long executionNanos, long second) {
            double rate = currentRate;
            boolean selected = selector.select(rate);
            return new Decision(selected, 0, rate, rate, "INV", "MONITORING", false);
        }

        @Override
        public synchronized RateUpdate onOperationsPerSecond(long second, long operationsPerSecond) {
            double previous = currentRate;
            if (operationsPerSecond == 0L) {
                return new RateUpdate(true, previous, previous, "", "ZERO_IGNORED",
                        "MONITORING", "", null);
            }
            minimumThroughput = Math.min(minimumThroughput, (double) operationsPerSecond);
            maximumThroughput = Math.max(maximumThroughput, (double) operationsPerSecond);
            double legacy = initialRate * ((maximumThroughput - (double) operationsPerSecond)
                    / (maximumThroughput - minimumThroughput));
            String status;
            double effective;
            if (Double.isNaN(legacy)) {
                effective = previous;
                status = "LEGACY_NAN_KEEP_PREVIOUS";
            } else {
                effective = legacy;
                status = "FORMULA";
                if (effective > initialRate) {
                    effective = initialRate;
                    status = "CLAMP_MAX";
                }
                if (effective < 0.01) {
                    effective = 0.01;
                    status = "CLAMP_MIN";
                }
                currentRate = effective;
            }
            if (telemetryConsole) {
                AgentLog.info("INV second=" + second + " operations=" + operationsPerSecond
                        + " legacy_rate=" + Double.toString(legacy)
                        + " effective_rate=" + Double.toString(effective)
                        + " status=" + status);
            }
            return new RateUpdate(true, previous, effective, Double.toString(legacy), status,
                    "MONITORING", "", null);
        }
    }

    /**
     * Rebuilds fixed-policy cycles from request completion seconds.
     *
     * The original workload may omit its top-level throughput callback for a
     * second in one policy even when that same second is an ADP marker in
     * another run.  Therefore fixed cycles cannot be closed only from the
     * policy-local throughput callback.  Requests are accumulated by benchmark
     * second and folded into the complete ADP marker list at shutdown.
     */
    final class CycleTrackingPolicy implements SamplingPolicy {
        private final SamplingPolicy delegate;
        private final List<Long> markers;
        private final Map<Long, FixedSecondData> bySecond =
                new TreeMap<Long, FixedSecondData>();

        CycleTrackingPolicy(SamplingPolicy delegate, List<Long> markers) {
            this.delegate = delegate;
            this.markers = markers;
            AgentLog.info("fixed cycle reconstruction enabled markers=" + markers.size());
        }

        @Override
        public synchronized Decision decide(String requestType, double value,
                                             long executionNanos, long second) {
            Decision decision = delegate.decide(
                    requestType, value, executionNanos, second);
            Long key = Long.valueOf(Math.max(0L, second));
            FixedSecondData data = bySecond.get(key);
            if (data == null) {
                data = new FixedSecondData();
                bySecond.put(key, data);
            }
            data.record(requestType, value, executionNanos, decision.sampled);
            return decision;
        }

        @Override
        public synchronized RateUpdate onOperationsPerSecond(
                long second, long operationsPerSecond) {
            return delegate.onOperationsPerSecond(second, operationsPerSecond);
        }

        @Override
        public synchronized List<CycleSnapshot> finishCycles(long second) {
            List<CycleSnapshot> result = new ArrayList<CycleSnapshot>();
            LegacyFrequencyDataSet population = new LegacyFrequencyDataSet();
            LegacyFrequencyDataSet sample = new LegacyFrequencyDataSet();
            MemoryMeans memory = new MemoryMeans();
            List<Map.Entry<Long, FixedSecondData>> entries =
                    new ArrayList<Map.Entry<Long, FixedSecondData>>(bySecond.entrySet());
            int entryIndex = 0;
            long previousMarker = 0L;
            int cycleIndex = 0;

            for (int markerIndex = 0; markerIndex < markers.size(); markerIndex++) {
                long marker = markers.get(markerIndex).longValue();
                if (marker > second) break;
                while (entryIndex < entries.size()
                        && entries.get(entryIndex).getKey().longValue() <= marker) {
                    FixedSecondData data = entries.get(entryIndex).getValue();
                    population.merge(data.population);
                    sample.merge(data.sample);
                    memory.merge(data.memory);
                    entryIndex += 1;
                }
                cycleIndex += 1;
                boolean finalCycle = markerIndex == markers.size() - 1;
                long elapsedMillis = Math.max(0L, marker - previousMarker) * 1000L;
                result.add(new CycleSnapshot(
                        cycleIndex, marker, finalCycle,
                        population.total(), sample.total(),
                        population.meanExecutionTime(), population.stdExecutionTime(),
                        sample.meanExecutionTime(), sample.stdExecutionTime(),
                        memory.selectedMeanAcrossTypes(), memory.denseMeanAcrossTypes(),
                        memory.selectedTypes(), memory.denseTypes(), elapsedMillis,
                        finalCycle ? "FINAL_MARKER_REBUILT" : "FIXED_MARKER_REBUILT"));
                population.clear();
                sample.clear();
                memory.clear();
                previousMarker = marker;
            }
            return result;
        }

        private static final class FixedSecondData {
            final LegacyFrequencyDataSet population = new LegacyFrequencyDataSet();
            final LegacyFrequencyDataSet sample = new LegacyFrequencyDataSet();
            final MemoryMeans memory = new MemoryMeans();

            void record(String requestType, double value, long executionNanos,
                        boolean selected) {
                population.add(requestType, (double) executionNanos);
                if (selected) sample.add(requestType, (double) executionNanos);
                memory.record(requestType, value, selected);
            }
        }
    }

    /** One independent OmniFlow controller per request type. */
    final class OmniFlowPolicy implements SamplingPolicy {
        private final AgentConfig config;
        private final Map<String, Controller> controllers = new ConcurrentHashMap<String, Controller>();

        OmniFlowPolicy(AgentConfig config) { this.config = config; }

        @Override
        public Decision decide(String requestType, double value, long executionNanos, long second) {
            Controller controller = controllers.get(requestType);
            if (controller == null) {
                Controller candidate = new Controller(config);
                Controller previous = controllers.putIfAbsent(requestType, candidate);
                controller = previous == null ? candidate : previous;
            }
            double signal;
            if ("heap_positive".equals(config.omniSignal)) {
                signal = Math.max(0.0, value);
            } else if ("duration_nanos".equals(config.omniSignal)) {
                signal = (double) executionNanos;
            } else {
                signal = value;
            }
            return controller.step(signal, value);
        }

        @Override
        public List<ControllerSnapshot> controllerSnapshots() {
            List<String> requestTypes = new ArrayList<String>(controllers.keySet());
            Collections.sort(requestTypes);
            List<ControllerSnapshot> result = new ArrayList<ControllerSnapshot>();
            for (String requestType : requestTypes) {
                Controller controller = controllers.get(requestType);
                if (controller != null) result.add(controller.snapshot(requestType, config.omniSignal));
            }
            return result;
        }
    }

    enum Status {
        INIT,
        NORMAL,
        OUTLIER,
        DRIFT_RESET
    }

    final class Tick {
        final Status status;
        final double zScore;

        Tick(Status status, double zScore) {
            this.status = status;
            this.zScore = zScore;
        }
    }

    /** Java translation of WindowedTracker + AdaptivePoller's linear mapping. */
    final class Controller {
        private static final double EPS = 1e-8;
        private static final double INV_SQRT_2PI = 1.0 / Math.sqrt(2.0 * Math.PI);

        private final AgentConfig c;
        private final Deque<Status> recentStatuses;
        private final Deque<Double> warmupStds;

        private boolean initialized;
        private double mean;
        private double variance;
        private double std;
        private int consecutiveOutliers;

        private double sigmaRef;
        private boolean sigmaRefAuto;
        private double urgency = 1.0;
        private double driftComponent;
        private double outlierComponent;
        private int stepsSinceSample;

        private long decisions;
        private long sampledDecisions;
        private long skippedDecisions;
        private long initCount;
        private long normalCount;
        private long outlierCount;
        private long driftResetCount;
        private long rawHeapPositive;
        private long rawHeapZero;
        private long rawHeapNegative;
        private long calibratedAtDecision = -1L;
        private long calibratedAtSample = -1L;
        private long intervalSum;
        private int observedMinInterval = Integer.MAX_VALUE;
        private int observedMaxInterval = Integer.MIN_VALUE;
        private double urgencySum;
        private final RunningStats sampledSignal = new RunningStats();

        Controller(AgentConfig config) {
            this.c = config;
            this.sigmaRef = config.sigmaRef;
            this.sigmaRefAuto = config.sigmaRef == 0.0;
            this.recentStatuses = new ArrayDeque<Status>(config.instabilityWindow);
            this.warmupStds = new ArrayDeque<Double>(config.warmup);
        }

        synchronized Decision step(double value, double rawHeapDelta) {
            decisions += 1L;
            if (rawHeapDelta > 0.0) rawHeapPositive += 1L;
            else if (rawHeapDelta < 0.0) rawHeapNegative += 1L;
            else rawHeapZero += 1L;

            int currentInterval = intervalFromUrgency(urgency);
            intervalSum += currentInterval;
            observedMinInterval = Math.min(observedMinInterval, currentInterval);
            observedMaxInterval = Math.max(observedMaxInterval, currentInterval);
            urgencySum += urgency;
            stepsSinceSample += 1;
            boolean first = !initialized;
            if (stepsSinceSample < currentInterval && !first) {
                skippedDecisions += 1L;
                return new Decision(false, currentInterval, 1.0 / currentInterval,
                        urgency, "SKIP", "MONITORING", false);
            }

            boolean wasAuto = sigmaRefAuto;
            Tick tick = updateTracker(value);
            sampledSignal.add(value);
            sampledDecisions += 1L;
            if (tick.status == Status.INIT) initCount += 1L;
            else if (tick.status == Status.NORMAL) normalCount += 1L;
            else if (tick.status == Status.OUTLIER) outlierCount += 1L;
            else if (tick.status == Status.DRIFT_RESET) driftResetCount += 1L;
            stepsSinceSample = 0;
            appendStatus(tick.status);
            updateUrgency(tick);
            if (wasAuto && !sigmaRefAuto && calibratedAtDecision < 0L) {
                calibratedAtDecision = decisions;
                calibratedAtSample = sampledDecisions;
            }
            return new Decision(true, currentInterval, 1.0 / currentInterval,
                    urgency, tick.status.name(), "MONITORING", false);
        }

        synchronized ControllerSnapshot snapshot(String requestType, String signalMode) {
            int finalInterval = intervalFromUrgency(urgency);
            RunningStats.Snapshot signal = sampledSignal.snapshot();
            return new ControllerSnapshot(
                    requestType, signalMode, decisions, sampledDecisions, skippedDecisions,
                    initCount, normalCount, outlierCount, driftResetCount,
                    rawHeapPositive, rawHeapZero, rawHeapNegative, !sigmaRefAuto,
                    calibratedAtDecision, calibratedAtSample,
                    observedMinInterval == Integer.MAX_VALUE ? finalInterval : observedMinInterval,
                    decisions == 0L ? 0.0 : (double) intervalSum / (double) decisions,
                    observedMaxInterval == Integer.MIN_VALUE ? finalInterval : observedMaxInterval,
                    decisions == 0L ? 0.0 : urgencySum / (double) decisions,
                    urgency, finalInterval, sigmaRef, std, signal.mean, signal.std);
        }

        private Tick updateTracker(double x) {
            if (!initialized) {
                mean = x;
                variance = square(Math.abs(x) * 0.1);
                std = Math.sqrt(variance);
                initialized = true;
                return new Tick(Status.INIT, 0.0);
            }

            double z = (x - mean) / (std + EPS);
            if (Math.abs(z) > c.outlierThreshold) {
                consecutiveOutliers += 1;
                if (consecutiveOutliers >= c.driftTolerance) {
                    mean = x;
                    variance = square(Math.abs(x) * 0.1);
                    std = Math.sqrt(variance);
                    consecutiveOutliers = 0;
                    return new Tick(Status.DRIFT_RESET, z);
                }
                return new Tick(Status.OUTLIER, z);
            }

            consecutiveOutliers = 0;
            double p = INV_SQRT_2PI * Math.exp(-0.5 * z * z);
            double alpha = c.alphaBase * (1.0 - c.beta * p);
            double oldMean = mean;
            mean = (1.0 - alpha) * mean + alpha * x;
            variance = (1.0 - alpha) * variance + alpha * square(x - oldMean);
            std = Math.sqrt(variance);
            return new Tick(Status.NORMAL, z);
        }

        private void appendStatus(Status status) {
            if (recentStatuses.size() >= c.instabilityWindow) {
                recentStatuses.removeFirst();
            }
            recentStatuses.addLast(status);
        }

        private void updateUrgency(Tick tick) {
            if (sigmaRefAuto) {
                if (tick.status == Status.NORMAL && std > 0.0) {
                    if (warmupStds.size() >= c.warmup) {
                        warmupStds.removeFirst();
                    }
                    warmupStds.addLast(std);
                    if (warmupStds.size() >= c.warmup) {
                        sigmaRef = warmupStds.getLast();
                        sigmaRefAuto = false;
                    }
                }
                urgency = 1.0;
                return;
            }

            if (c.sigmaRefAdapt > 0.0 && std > 0.0) {
                sigmaRef = (1.0 - c.sigmaRefAdapt) * sigmaRef + c.sigmaRefAdapt * std;
            }

            double varianceComponent = clip((std / (sigmaRef + EPS) - 1.0) * c.varianceSensitivity);

            int unstable = 0;
            for (Status status : recentStatuses) {
                if (status == Status.OUTLIER || status == Status.DRIFT_RESET) {
                    unstable += 1;
                }
            }
            double instabilityComponent = recentStatuses.isEmpty()
                    ? 0.0
                    : clip(((double) unstable / (double) recentStatuses.size()) * c.instabilityWeight);

            if (tick.status == Status.DRIFT_RESET) {
                driftComponent = clip(c.driftBoost);
            } else {
                driftComponent = clip(driftComponent * c.driftDecay);
            }

            if (tick.status == Status.OUTLIER || tick.status == Status.DRIFT_RESET) {
                double kick = Math.min(1.0,
                        (Math.abs(tick.zScore) - c.outlierThreshold) / c.outlierThreshold);
                outlierComponent = Math.max(outlierComponent, kick);
            } else {
                outlierComponent = clip(outlierComponent * c.outlierDecay);
            }

            double raw = Math.max(Math.max(varianceComponent, instabilityComponent),
                    Math.max(driftComponent, outlierComponent));
            raw = clip(raw);
            urgency = c.urgencySmoothing * urgency + (1.0 - c.urgencySmoothing) * raw;
            if (raw < c.cooldownThreshold) {
                urgency *= (1.0 - c.cooldownRate);
            }
            urgency = clip(urgency);
        }

        private int intervalFromUrgency(double value) {
            double u = clip(value);
            double interval = c.maxInterval - (c.maxInterval - c.minInterval) * u;
            int rounded = (int) Math.rint(interval);
            if (rounded < c.minInterval) return c.minInterval;
            if (rounded > c.maxInterval) return c.maxInterval;
            return rounded;
        }

        private static double square(double x) {
            return x * x;
        }

        private static double clip(double x) {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return x;
        }
    }
}
