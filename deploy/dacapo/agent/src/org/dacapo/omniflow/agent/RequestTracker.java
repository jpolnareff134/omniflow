package org.dacapo.omniflow.agent;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/** Measures per-request heap deltas and delegates selection to a policy. */
public final class RequestTracker {
    private static volatile AgentConfig config;
    private static volatile SamplingPolicy policy;
    private static volatile PrintWriter rowWriter;
    private static volatile long startNano;
    private static final AtomicLong workloadStartNano = new AtomicLong(-1L);
    private static final AtomicLong lastRequestSecond = new AtomicLong(-1L);
    private static final AtomicLong maxObservedRequestSecond = new AtomicLong(-1L);
    private static final AtomicLong postHorizonCapped = new AtomicLong();

    private static final ThreadLocal<Integer> DEPTH = new ThreadLocal<Integer>() {
        @Override protected Integer initialValue() { return Integer.valueOf(0); }
    };
    private static final ThreadLocal<Request> CURRENT = new ThreadLocal<Request>();

    private static final AtomicBoolean acceptingEvents = new AtomicBoolean(true);
    private static final AtomicInteger activeRecorders = new AtomicInteger();
    private static final AtomicLong eventSequence = new AtomicLong();
    private static final AtomicLong total = new AtomicLong();
    private static final AtomicLong sampled = new AtomicLong();
    private static final ConcurrentHashMap<String, Stats> stats = new ConcurrentHashMap<String, Stats>();
    private static final ConcurrentHashMap<Long, BucketStats> telemetry = new ConcurrentHashMap<Long, BucketStats>();
    private static final List<SamplingPolicy.CycleSnapshot> cycles =
            Collections.synchronizedList(new ArrayList<SamplingPolicy.CycleSnapshot>());

    private static final AtomicLong operationIntervalCount = new AtomicLong();
    private static final AtomicBoolean operationsStarted = new AtomicBoolean();
    private static final AtomicBoolean operationsRunning = new AtomicBoolean();
    private static final AtomicBoolean operationsFlushed = new AtomicBoolean();
    private static volatile Thread operationsThread;
    private static long lastOperationsTelemetrySecond = -1L;

    private RequestTracker() { }

    public static synchronized void initialize(AgentConfig cfg) {
        if (config != null) return;
        config = cfg;
        policy = SamplingPolicy.create(cfg);
        startNano = System.nanoTime();
        if (cfg.writeRows) {
            try {
                File output = prepareOutputFile(cfg.output);
                rowWriter = new PrintWriter(new BufferedWriter(new FileWriter(output, false), 1 << 20));
                rowWriter.println(
                        "event_index,elapsed_nanos,second,request_type,delta_heap,sampled,"
                        + "interval,effective_rate,urgency,status,baseline_state,cycle_marker");
            } catch (IOException e) {
                throw new RuntimeException("Cannot open output: " + cfg.output, e);
            }
        }
        AgentLog.info("policy=" + cfg.policy
                + " output=" + cfg.output
                + " write_rows=" + cfg.writeRows
                + " uniform_rate=" + cfg.uniformRate
                + " seed=" + cfg.seed
                + " omni_signal=" + cfg.omniSignal
                + " cycle_length_millis=" + cfg.cycleLengthMillis
                + " exact_benchmark_seconds=" + cfg.exactBenchmarkSeconds
                + " skip_cassandra_native_check=" + cfg.skipCassandraNativeCheck
                + " final_marker_second=" + cfg.finalMarkerSecond
                + " max_interval=" + cfg.maxInterval);
    }

    public static String requestType(String methodId, Object firstArgument) {
        return methodId + "|" + String.valueOf(firstArgument);
    }

    /** Exact benchmark-provided throughput bucket (second, operations). */
    public static synchronized void operationsPerSecond(int second, int operations) {
        if (config == null || policy == null || !acceptingEvents.get()) return;
        if (second < 0 || operations < 0) return;
        activeRecorders.incrementAndGet();
        try {
            if (!acceptingEvents.get()) return;
            long sec = (long) second;
            if (sec > lastOperationsTelemetrySecond) lastOperationsTelemetrySecond = sec;
            SamplingPolicy.RateUpdate update = policy.onOperationsPerSecond(sec, (long) operations);
            if (update != null && update.cycle != null) cycles.add(update.cycle);
            bucketFor(sec).recordOperations((long) operations, update);
        } finally {
            activeRecorders.decrementAndGet();
        }
    }

    public static void operationCompleted() {
        if (config == null || policy == null || !acceptingEvents.get()) return;
        activeRecorders.incrementAndGet();
        try {
            if (!acceptingEvents.get()) return;
            markWorkloadStart(System.nanoTime());
            operationIntervalCount.incrementAndGet();
            if (!operationsStarted.get() && operationsStarted.compareAndSet(false, true)) {
                startOperationsReporter();
            }
        } finally {
            activeRecorders.decrementAndGet();
        }
    }

    public static void startRequest(String requestType) {
        if (config == null || !acceptingEvents.get()) return;
        int depth = DEPTH.get().intValue();
        DEPTH.set(Integer.valueOf(depth + 1));
        if (depth != 0) return;
        long now = System.nanoTime();
        markWorkloadStart(now);
        CURRENT.set(new Request(requestType, usedHeap(), now));
    }

    public static void endRequest() {
        if (config == null) return;
        int depth = DEPTH.get().intValue();
        if (depth <= 0) return;
        DEPTH.set(Integer.valueOf(depth - 1));
        if (depth != 1) return;

        Request request = CURRENT.get();
        CURRENT.remove();
        if (request == null || !acceptingEvents.get()) return;

        activeRecorders.incrementAndGet();
        try {
            if (!acceptingEvents.get()) return;

            long now = System.nanoTime();
            long epoch = workloadStartNano.get();
            long elapsedNanos = Math.max(0L, now - (epoch < 0L ? startNano : epoch));
            long observedSecond = elapsedNanos / 1000000000L;
            updateMaximum(maxObservedRequestSecond, observedSecond);
            long second = observedSecond;
            if (config.finalMarkerSecond >= 0L && second > config.finalMarkerSecond) {
                /*
                 * An original fixed-duration benchmark harness may resume from a long
                 * JVM pause after the nominal schedule has ended.  Requests that
                 * were already in flight are still completed before stopIteration.
                 * Keep them in the final monitoring cycle instead of creating a
                 * policy-local marker beyond the shared 1,780-second horizon.
                 */
                postHorizonCapped.incrementAndGet();
                second = config.finalMarkerSecond;
            }
            updateMaximum(lastRequestSecond, second);
            long eventIndex = eventSequence.getAndIncrement();
            long delta = usedHeap() - request.startHeap;
            long executionNanos = Math.max(0L, now - request.startNano);
            SamplingPolicy.Decision decision = policy.decide(
                    request.requestType, (double) delta, executionNanos, second);

            total.incrementAndGet();
            Stats typeStats = stats.get(request.requestType);
            if (typeStats == null) {
                Stats candidate = new Stats();
                Stats previous = stats.putIfAbsent(request.requestType, candidate);
                typeStats = previous == null ? candidate : previous;
            }
            typeStats.record(delta, decision.sampled);
            if (decision.sampled) sampled.incrementAndGet();

            bucketFor(second).record(decision);

            PrintWriter writer = rowWriter;
            if (writer != null && decision.sampled) {
                synchronized (writer) {
                    writer.print(eventIndex);
                    writer.print(',');
                    writer.print(elapsedNanos);
                    writer.print(',');
                    writer.print(second);
                    writer.print(',');
                    writer.print(csv(request.requestType));
                    writer.print(',');
                    writer.print(delta);
                    writer.print(",true,");
                    writer.print(decision.interval);
                    writer.print(',');
                    writer.print(decision.rate);
                    writer.print(',');
                    writer.print(decision.urgency);
                    writer.print(',');
                    writer.print(decision.status);
                    writer.print(',');
                    writer.print(decision.baselineState);
                    writer.print(',');
                    writer.println(decision.cycleMarker);
                }
            }
        } finally {
            activeRecorders.decrementAndGet();
        }
    }

    public static synchronized void shutdown() {
        acceptingEvents.set(false);
        waitForActiveRecorders();
        flushOperations();
        PrintWriter writer = rowWriter;
        if (writer != null) {
            writer.flush();
            writer.close();
            rowWriter = null;
        }
        if (config == null) return;
        try {
            long finalSecond = Math.max(lastOperationsTelemetrySecond, lastRequestSecond.get());
            if (config.finalMarkerSecond >= 0L) finalSecond = Math.max(finalSecond, config.finalMarkerSecond);
            List<SamplingPolicy.CycleSnapshot> finalCycles =
                    policy.finishCycles(Math.max(0L, finalSecond));
            for (SamplingPolicy.CycleSnapshot cycle : finalCycles) {
                SamplingPolicy.CycleSnapshot stored = storeFinalCycle(cycle);
                bucketFor(stored.markerSecond).recordCycle(stored);
            }
            writeSummary(config.output + ".summary.csv");
            writeControllerDiagnostics(config.output + ".controller.csv");
            writeTelemetry(config.output + ".telemetry.csv");
            writeCycles(config.output + ".cycles.csv");
            writeMarkers(config.output + ".markers.txt");
        } catch (RuntimeException error) {
            AgentLog.error(error.getMessage(), error, config.verbose);
        }
        long n = total.get();
        long k = sampled.get();
        AgentLog.info("total=" + n + " sampled=" + k
                + " ratio=" + (n == 0 ? 0.0 : (double) k / (double) n)
                + " cycles=" + cycles.size()
                + " post_horizon_capped=" + postHorizonCapped.get()
                + " max_observed_request_second=" + maxObservedRequestSecond.get());
    }

    /**
     * Store a shutdown cycle without emitting the same marker twice.
     *
     * ADP may satisfy its readiness test at the exact second on which the JVM
     * is terminated.  The normal throughput callback then records a non-final
     * cycle and {@code finish()} sees a small post-reset fragment at that same
     * second.  The paper marker list must be strictly increasing, so represent
     * both fragments as one final marker instead of appending a duplicate.
     */
    private static SamplingPolicy.CycleSnapshot storeFinalCycle(
            SamplingPolicy.CycleSnapshot candidate) {
        synchronized (cycles) {
            int size = cycles.size();
            if (candidate.finalCycle && size > 0) {
                SamplingPolicy.CycleSnapshot previous = cycles.get(size - 1);
                if (previous.markerSecond == candidate.markerSecond) {
                    SamplingPolicy.CycleSnapshot merged = mergeAsFinal(previous, candidate);
                    cycles.set(size - 1, merged);
                    return merged;
                }
            }
            cycles.add(candidate);
            return candidate;
        }
    }

    private static SamplingPolicy.CycleSnapshot mergeAsFinal(
            SamplingPolicy.CycleSnapshot previous,
            SamplingPolicy.CycleSnapshot tail) {
        long population = previous.populationSize + tail.populationSize;
        long sample = previous.sampleSize + tail.sampleSize;
        double populationMean = pooledMean(
                previous.populationSize, previous.populationMeanNanos,
                tail.populationSize, tail.populationMeanNanos);
        double populationStd = pooledStd(
                previous.populationSize, previous.populationMeanNanos,
                previous.populationStdNanos,
                tail.populationSize, tail.populationMeanNanos,
                tail.populationStdNanos, populationMean);
        double sampleMean = pooledMean(
                previous.sampleSize, previous.sampleMeanNanos,
                tail.sampleSize, tail.sampleMeanNanos);
        double sampleStd = pooledStd(
                previous.sampleSize, previous.sampleMeanNanos,
                previous.sampleStdNanos,
                tail.sampleSize, tail.sampleMeanNanos,
                tail.sampleStdNanos, sampleMean);
        double denseMemory = finite(previous.denseMemoryMeanBytes)
                ? previous.denseMemoryMeanBytes : tail.denseMemoryMeanBytes;
        int denseTypes = Math.max(previous.denseMemoryTypes, tail.denseMemoryTypes);
        return new SamplingPolicy.CycleSnapshot(
                previous.cycleIndex,
                previous.markerSecond,
                true,
                population,
                sample,
                populationMean,
                populationStd,
                sampleMean,
                sampleStd,
                Double.NaN,
                denseMemory,
                0,
                denseTypes,
                previous.elapsedMillis + tail.elapsedMillis,
                "FINAL_MARKER_MERGED");
    }

    private static double pooledMean(long firstCount, double firstMean,
                                     long secondCount, double secondMean) {
        long totalCount = firstCount + secondCount;
        if (totalCount == 0L) return Double.NaN;
        double first = finite(firstMean) ? firstMean : 0.0;
        double second = finite(secondMean) ? secondMean : 0.0;
        return ((double) firstCount * first + (double) secondCount * second)
                / (double) totalCount;
    }

    private static double pooledStd(long firstCount, double firstMean, double firstStd,
                                    long secondCount, double secondMean, double secondStd,
                                    double pooledMean) {
        long totalCount = firstCount + secondCount;
        if (totalCount <= 1L || !finite(pooledMean)) return 0.0;
        double firstVariance = finite(firstStd) ? firstStd * firstStd : 0.0;
        double secondVariance = finite(secondStd) ? secondStd * secondStd : 0.0;
        double firstDelta = finite(firstMean) ? firstMean - pooledMean : 0.0;
        double secondDelta = finite(secondMean) ? secondMean - pooledMean : 0.0;
        double sumSquares = Math.max(0L, firstCount - 1L) * firstVariance
                + Math.max(0L, secondCount - 1L) * secondVariance
                + (double) firstCount * firstDelta * firstDelta
                + (double) secondCount * secondDelta * secondDelta;
        return Math.sqrt(sumSquares / (double) (totalCount - 1L));
    }

    private static boolean finite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }


    private static void waitForActiveRecorders() {
        long deadline = System.nanoTime() + 5000000000L;
        while (activeRecorders.get() != 0 && System.nanoTime() < deadline) {
            try {
                Thread.sleep(1L);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        int remaining = activeRecorders.get();
        if (remaining != 0) {
            AgentLog.info("shutdown snapshot continued with active_recorders=" + remaining);
        }
    }

    private static void writeSummary(String path) {
        try {
            File output = prepareOutputFile(path);
            PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(output, false)));
            out.println(
                    "request_type,total_requests,selected_requests,"
                    + "positive_total_count,positive_total_sum_bytes,positive_total_mean_bytes,"
                    + "positive_selected_count,positive_selected_sum_bytes,positive_selected_mean_bytes");
            Map<String, Stats> sorted = new TreeMap<String, Stats>(stats);
            for (Map.Entry<String, Stats> entry : sorted.entrySet()) {
                Stats.Snapshot s = entry.getValue().snapshot();
                out.print(csv(entry.getKey()));
                out.print(','); out.print(s.totalRequests);
                out.print(','); out.print(s.selectedRequests);
                out.print(','); out.print(s.positiveTotalCount);
                out.print(','); out.print(s.positiveTotalSum);
                out.print(','); out.print(meanOrEmpty(s.positiveTotalCount, s.positiveTotalSum));
                out.print(','); out.print(s.positiveSelectedCount);
                out.print(','); out.print(s.positiveSelectedSum);
                out.print(','); out.println(meanOrEmpty(s.positiveSelectedCount, s.positiveSelectedSum));
            }
            out.print("__overall__,");
            out.print(total.get());
            out.print(',');
            out.print(sampled.get());
            out.println(",,,,,,");
            out.close();
        } catch (IOException e) {
            throw new RuntimeException("Cannot write summary: " + path, e);
        }
    }

    private static void writeControllerDiagnostics(String path) {
        List<SamplingPolicy.ControllerSnapshot> snapshots = policy.controllerSnapshots();
        if (snapshots == null || snapshots.isEmpty()) return;
        try {
            File output = prepareOutputFile(path);
            PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(output, false)));
            out.println("request_type,signal_mode,decisions,sampled,skipped,sampling_ratio,"
                    + "init_count,normal_count,outlier_count,drift_reset_count,"
                    + "raw_heap_positive,raw_heap_zero,raw_heap_negative,positive_heap_ratio,"
                    + "calibrated,calibrated_at_decision,calibrated_at_sample,"
                    + "min_interval,mean_interval,max_interval,mean_urgency,final_urgency,"
                    + "final_interval,sigma_ref,tracker_std,sampled_signal_mean,sampled_signal_std");
            for (SamplingPolicy.ControllerSnapshot snapshot : snapshots) {
                out.print(csv(snapshot.requestType));
                out.print(','); out.print(snapshot.signalMode);
                out.print(','); out.print(snapshot.decisions);
                out.print(','); out.print(snapshot.sampled);
                out.print(','); out.print(snapshot.skipped);
                out.print(','); out.print(snapshot.decisions == 0L ? 0.0
                        : (double) snapshot.sampled / (double) snapshot.decisions);
                out.print(','); out.print(snapshot.initCount);
                out.print(','); out.print(snapshot.normalCount);
                out.print(','); out.print(snapshot.outlierCount);
                out.print(','); out.print(snapshot.driftResetCount);
                out.print(','); out.print(snapshot.rawHeapPositive);
                out.print(','); out.print(snapshot.rawHeapZero);
                out.print(','); out.print(snapshot.rawHeapNegative);
                out.print(','); out.print(snapshot.decisions == 0L ? 0.0
                        : (double) snapshot.rawHeapPositive / (double) snapshot.decisions);
                out.print(','); out.print(snapshot.calibrated);
                out.print(','); out.print(snapshot.calibratedAtDecision);
                out.print(','); out.print(snapshot.calibratedAtSample);
                out.print(','); out.print(snapshot.minInterval);
                out.print(','); out.print(snapshot.meanInterval);
                out.print(','); out.print(snapshot.maxInterval);
                out.print(','); out.print(snapshot.meanUrgency);
                out.print(','); out.print(snapshot.finalUrgency);
                out.print(','); out.print(snapshot.finalInterval);
                out.print(','); out.print(snapshot.sigmaRef);
                out.print(','); out.print(snapshot.trackerStd);
                out.print(','); out.print(snapshot.sampledSignalMean);
                out.print(','); out.println(snapshot.sampledSignalStd);
            }
            out.close();
        } catch (IOException error) {
            throw new RuntimeException("Cannot write controller diagnostics: " + path, error);
        }
    }

    private static void writeTelemetry(String path) {
        try {
            File output = prepareOutputFile(path);
            PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(output, false)));
            out.println(
                    "second,policy,agent_requests_per_second,selected_requests_per_second,"
                    + "top_level_operations_per_second,observed_sampling_ratio,"
                    + "current_rate,min_rate,max_rate,next_rate,"
                    + "population_size,sample_size,baseline_state,cycle_marker,cycle_index,"
                    + "cycle_readiness,legacy_computed_rate,rate_update_status,readiness_state");
            Map<Long, BucketStats> sorted = new TreeMap<Long, BucketStats>(telemetry);
            long cumulativePopulation = 0L;
            long cumulativeSample = 0L;
            for (Map.Entry<Long, BucketStats> entry : sorted.entrySet()) {
                BucketStats.Snapshot s = entry.getValue().snapshot();
                cumulativePopulation += s.requests;
                cumulativeSample += s.selected;
                out.print(entry.getKey().longValue());
                out.print(','); out.print(config.policy);
                double currentRate = s.requests == 0L
                        ? (s.hasRateUpdate ? s.previousRate : 0.0)
                        : s.rateSum / (double) s.requests;
                double minimumRate = s.requests == 0L
                        ? currentRate : s.minRate;
                double maximumRate = s.requests == 0L
                        ? currentRate : s.maxRate;
                double nextRate = s.hasRateUpdate ? s.nextRate
                        : (s.requests == 0L ? currentRate : s.lastRate);
                out.print(','); out.print(s.requests);
                out.print(','); out.print(s.selected);
                out.print(','); out.print(s.topLevelOperations);
                out.print(','); out.print(s.requests == 0L ? 0.0 : (double) s.selected / (double) s.requests);
                out.print(','); out.print(currentRate);
                out.print(','); out.print(minimumRate);
                out.print(','); out.print(maximumRate);
                out.print(','); out.print(nextRate);
                out.print(','); out.print(cumulativePopulation);
                out.print(','); out.print(cumulativeSample);
                out.print(','); out.print(s.baselineState);
                out.print(','); out.print(s.cycleMarker);
                out.print(','); out.print(s.cycleIndex);
                out.print(','); out.print(csv(s.cycleReadiness));
                out.print(','); out.print(s.legacyComputedRate);
                out.print(','); out.print(s.rateUpdateStatus);
                out.print(','); out.println(csv(s.readinessState));
            }
            out.close();
        } catch (IOException e) {
            throw new RuntimeException("Cannot write telemetry: " + path, e);
        }
    }

    private static void writeCycles(String path) {
        try {
            File output = prepareOutputFile(path);
            PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(output, false)));
            out.println("cycle_index,marker_second,final_cycle,population_size,sample_size,"
                    + "population_mean_nanos,population_std_nanos,sample_mean_nanos,sample_std_nanos,"
                    + "selected_memory_mean_bytes,dense_memory_mean_bytes,selected_memory_types,"
                    + "dense_memory_types,elapsed_millis,readiness");
            synchronized (cycles) {
                for (SamplingPolicy.CycleSnapshot cycle : cycles) {
                    out.print(cycle.cycleIndex);
                    out.print(','); out.print(cycle.markerSecond);
                    out.print(','); out.print(cycle.finalCycle);
                    out.print(','); out.print(cycle.populationSize);
                    out.print(','); out.print(cycle.sampleSize);
                    out.print(','); out.print(numberOrEmpty(cycle.populationMeanNanos));
                    out.print(','); out.print(numberOrEmpty(cycle.populationStdNanos));
                    out.print(','); out.print(numberOrEmpty(cycle.sampleMeanNanos));
                    out.print(','); out.print(numberOrEmpty(cycle.sampleStdNanos));
                    out.print(','); out.print(numberOrEmpty(cycle.selectedMemoryMeanBytes));
                    out.print(','); out.print(numberOrEmpty(cycle.denseMemoryMeanBytes));
                    out.print(','); out.print(cycle.selectedMemoryTypes);
                    out.print(','); out.print(cycle.denseMemoryTypes);
                    out.print(','); out.print(cycle.elapsedMillis);
                    out.print(','); out.println(csv(cycle.readiness));
                }
            }
            out.close();
        } catch (IOException error) {
            throw new RuntimeException("Cannot write cycles: " + path, error);
        }
    }

    private static void writeMarkers(String path) {
        try {
            File output = prepareOutputFile(path);
            PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(output, false)));
            synchronized (cycles) {
                for (SamplingPolicy.CycleSnapshot cycle : cycles) {
                    out.println(cycle.markerSecond);
                }
            }
            out.close();
        } catch (IOException error) {
            throw new RuntimeException("Cannot write markers: " + path, error);
        }
    }

    private static String numberOrEmpty(double value) {
        return Double.isNaN(value) || Double.isInfinite(value) ? "" : Double.toString(value);
    }

    private static void markWorkloadStart(long now) {
        workloadStartNano.compareAndSet(-1L, now);
    }

    private static void updateMaximum(AtomicLong target, long value) {
        while (true) {
            long current = target.get();
            if (value <= current || target.compareAndSet(current, value)) return;
        }
    }

    private static BucketStats bucketFor(long second) {
        Long key = Long.valueOf(second);
        BucketStats bucket = telemetry.get(key);
        if (bucket == null) {
            BucketStats candidate = new BucketStats();
            BucketStats previous = telemetry.putIfAbsent(key, candidate);
            bucket = previous == null ? candidate : previous;
        }
        return bucket;
    }

    private static void startOperationsReporter() {
        operationsRunning.set(true);
        Thread reporter = new Thread(new Runnable() {
            @Override
            public void run() {
                while (operationsRunning.get()) {
                    try {
                        Thread.sleep(1000L);
                    } catch (InterruptedException ignored) {
                        // Shutdown interrupts the reporter so the final partial
                        // interval can be drained immediately by the hook.
                    }
                    if (!operationsRunning.get()) break;
                    drainOperationsInterval(true);
                }
            }
        }, "omniflow-throughput-reporter");
        reporter.setDaemon(true);
        operationsThread = reporter;
        reporter.start();
    }

    private static synchronized void drainOperationsInterval(boolean completedSecond) {
        long count = operationIntervalCount.getAndSet(0L);
        if (count == 0L) return;
        long epoch = workloadStartNano.get();
        long wallSecond = Math.max(0L, System.nanoTime() - (epoch < 0L ? startNano : epoch)) / 1000000000L;
        long candidate = completedSecond ? Math.max(0L, wallSecond - 1L) : wallSecond;
        long second = Math.max(candidate, lastOperationsTelemetrySecond + 1L);
        lastOperationsTelemetrySecond = second;
        SamplingPolicy.RateUpdate update = policy.onOperationsPerSecond(second, count);
        if (update != null && update.cycle != null) cycles.add(update.cycle);
        bucketFor(second).recordOperations(count, update);
    }

    private static void flushOperations() {
        if (!operationsFlushed.compareAndSet(false, true)) return;
        operationsRunning.set(false);
        Thread reporter = operationsThread;
        if (reporter != null) {
            reporter.interrupt();
            try {
                reporter.join(5000L);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        drainOperationsInterval(false);
    }

    private static String meanOrEmpty(long count, double sum) {
        return count == 0 ? "" : Double.toString(sum / count);
    }

    private static File prepareOutputFile(String path) throws IOException {
        File output = new File(path).getAbsoluteFile();
        File parent = output.getParentFile();
        if (parent != null && !parent.isDirectory()) {
            if (!parent.mkdirs() && !parent.isDirectory()) {
                throw new IOException("Cannot create output directory: " + parent);
            }
        }
        return output;
    }

    private static long usedHeap() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private static String csv(String value) {
        if (value == null) return "";
        boolean quote = value.indexOf(',') >= 0 || value.indexOf('"') >= 0
                || value.indexOf('\n') >= 0 || value.indexOf('\r') >= 0;
        if (!quote) return value;
        return '"' + value.replace("\"", "\"\"") + '"';
    }

    private static final class Request {
        final String requestType;
        final long startHeap;
        final long startNano;
        Request(String requestType, long startHeap, long startNano) {
            this.requestType = requestType;
            this.startHeap = startHeap;
            this.startNano = startNano;
        }
    }

    private static final class Stats {
        private long totalRequests;
        private long selectedRequests;
        private long positiveTotalCount;
        private double positiveTotalSum;
        private long positiveSelectedCount;
        private double positiveSelectedSum;

        synchronized void record(long delta, boolean selected) {
            totalRequests += 1;
            if (delta > 0) {
                positiveTotalCount += 1;
                positiveTotalSum += (double) delta;
            }
            if (selected) {
                selectedRequests += 1;
                if (delta > 0) {
                    positiveSelectedCount += 1;
                    positiveSelectedSum += (double) delta;
                }
            }
        }

        synchronized Snapshot snapshot() {
            return new Snapshot(
                    totalRequests,
                    selectedRequests,
                    positiveTotalCount,
                    positiveTotalSum,
                    positiveSelectedCount,
                    positiveSelectedSum);
        }

        static final class Snapshot {
            final long totalRequests;
            final long selectedRequests;
            final long positiveTotalCount;
            final double positiveTotalSum;
            final long positiveSelectedCount;
            final double positiveSelectedSum;

            Snapshot(
                    long totalRequests,
                    long selectedRequests,
                    long positiveTotalCount,
                    double positiveTotalSum,
                    long positiveSelectedCount,
                    double positiveSelectedSum) {
                this.totalRequests = totalRequests;
                this.selectedRequests = selectedRequests;
                this.positiveTotalCount = positiveTotalCount;
                this.positiveTotalSum = positiveTotalSum;
                this.positiveSelectedCount = positiveSelectedCount;
                this.positiveSelectedSum = positiveSelectedSum;
            }
        }
    }

    private static final class BucketStats {
        private long requests;
        private long selected;
        private long topLevelOperations;
        private double rateSum;
        private double minRate = Double.POSITIVE_INFINITY;
        private double maxRate = Double.NEGATIVE_INFINITY;
        private double lastRate;
        private String baselineState = "MONITORING";
        private boolean cycleMarker;
        private boolean hasRateUpdate;
        private double previousRate;
        private double nextRate;
        private String legacyComputedRate = "";
        private String rateUpdateStatus = "NOT_APPLICABLE";
        private String readinessState = "";
        private int cycleIndex;
        private String cycleReadiness = "";

        synchronized void record(SamplingPolicy.Decision decision) {
            requests += 1L;
            if (decision.sampled) selected += 1L;
            rateSum += decision.rate;
            minRate = Math.min(minRate, decision.rate);
            maxRate = Math.max(maxRate, decision.rate);
            lastRate = decision.rate;
            baselineState = decision.baselineState;
            cycleMarker = cycleMarker || decision.cycleMarker;
        }

        synchronized void recordOperations(
                long operations, SamplingPolicy.RateUpdate update) {
            topLevelOperations += operations;
            if (update != null) {
                baselineState = update.baselineState;
                if (update.applicable) {
                    hasRateUpdate = true;
                    previousRate = update.previousRate;
                    nextRate = update.effectiveRate;
                    legacyComputedRate = update.legacyComputedRate;
                    rateUpdateStatus = update.status;
                }
                readinessState = update.readinessState;
                if (update.cycle != null) {
                    cycleMarker = true;
                    cycleIndex = update.cycle.cycleIndex;
                    cycleReadiness = update.cycle.readiness;
                }
            }
        }

        synchronized void recordCycle(SamplingPolicy.CycleSnapshot cycle) {
            cycleMarker = true;
            cycleIndex = cycle.cycleIndex;
            cycleReadiness = cycle.readiness;
        }

        synchronized Snapshot snapshot() {
            return new Snapshot(
                    requests,
                    selected,
                    topLevelOperations,
                    rateSum,
                    minRate,
                    maxRate,
                    lastRate,
                    baselineState,
                    cycleMarker,
                    hasRateUpdate,
                    previousRate,
                    nextRate,
                    legacyComputedRate,
                    rateUpdateStatus,
                    readinessState,
                    cycleIndex,
                    cycleReadiness);
        }

        static final class Snapshot {
            final long requests;
            final long selected;
            final long topLevelOperations;
            final double rateSum;
            final double minRate;
            final double maxRate;
            final double lastRate;
            final String baselineState;
            final boolean cycleMarker;
            final boolean hasRateUpdate;
            final double previousRate;
            final double nextRate;
            final String legacyComputedRate;
            final String rateUpdateStatus;
            final String readinessState;
            final int cycleIndex;
            final String cycleReadiness;

            Snapshot(long requests, long selected, long topLevelOperations,
                     double rateSum, double minRate, double maxRate, double lastRate,
                     String baselineState, boolean cycleMarker, boolean hasRateUpdate,
                     double previousRate, double nextRate, String legacyComputedRate,
                     String rateUpdateStatus, String readinessState, int cycleIndex, String cycleReadiness) {
                this.requests = requests;
                this.selected = selected;
                this.topLevelOperations = topLevelOperations;
                this.rateSum = rateSum;
                this.minRate = minRate;
                this.maxRate = maxRate;
                this.lastRate = lastRate;
                this.baselineState = baselineState;
                this.cycleMarker = cycleMarker;
                this.hasRateUpdate = hasRateUpdate;
                this.previousRate = previousRate;
                this.nextRate = nextRate;
                this.legacyComputedRate = legacyComputedRate;
                this.rateUpdateStatus = rateUpdateStatus;
                this.readinessState = readinessState;
                this.cycleIndex = cycleIndex;
                this.cycleReadiness = cycleReadiness;
            }
        }
    }

}
