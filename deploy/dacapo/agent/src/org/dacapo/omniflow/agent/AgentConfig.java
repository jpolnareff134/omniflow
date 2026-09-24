package org.dacapo.omniflow.agent;

import java.util.Locale;

/** Agent configuration parsed from key=value pairs separated by commas. */
public final class AgentConfig {
    public final String output;
    public final String policy;
    public final boolean writeRows;
    public final boolean verbose;
    public final boolean disableOriginalSampler;
    public final boolean skipH2Reset;
    public final boolean skipCassandraNativeCheck;
    public final boolean telemetryConsole;
    public final boolean exactBenchmarkSeconds;
    public final long finalMarkerSecond;
    public final String cycleMarkersFile;
    public final long cycleLengthMillis;
    public final double uniformRate;
    public final long seed;
    public final String omniSignal;

    public final int minInterval;
    public final int maxInterval;
    public final double alphaBase;
    public final double beta;
    public final double outlierThreshold;
    public final int driftTolerance;
    public final double varianceSensitivity;
    public final double sigmaRef;
    public final int warmup;
    public final double sigmaRefAdapt;
    public final int instabilityWindow;
    public final double instabilityWeight;
    public final double driftBoost;
    public final double driftDecay;
    public final double cooldownRate;
    public final double urgencySmoothing;
    public final double cooldownThreshold;
    public final double outlierDecay;

    private AgentConfig(
            String output,
            String policy,
            boolean writeRows,
            boolean verbose,
            boolean disableOriginalSampler,
            boolean skipH2Reset,
            boolean skipCassandraNativeCheck,
            boolean telemetryConsole,
            boolean exactBenchmarkSeconds,
            long finalMarkerSecond,
            String cycleMarkersFile,
            long cycleLengthMillis,
            double uniformRate,
            long seed,
            String omniSignal,
            int minInterval,
            int maxInterval,
            double alphaBase,
            double beta,
            double outlierThreshold,
            int driftTolerance,
            double varianceSensitivity,
            double sigmaRef,
            int warmup,
            double sigmaRefAdapt,
            int instabilityWindow,
            double instabilityWeight,
            double driftBoost,
            double driftDecay,
            double cooldownRate,
            double urgencySmoothing,
            double cooldownThreshold,
            double outlierDecay) {
        this.output = output;
        this.policy = policy;
        this.writeRows = writeRows;
        this.verbose = verbose;
        this.disableOriginalSampler = disableOriginalSampler;
        this.skipH2Reset = skipH2Reset;
        this.skipCassandraNativeCheck = skipCassandraNativeCheck;
        this.telemetryConsole = telemetryConsole;
        this.exactBenchmarkSeconds = exactBenchmarkSeconds;
        this.finalMarkerSecond = finalMarkerSecond;
        this.cycleMarkersFile = cycleMarkersFile;
        this.cycleLengthMillis = cycleLengthMillis;
        this.uniformRate = uniformRate;
        this.seed = seed;
        this.omniSignal = omniSignal;
        this.minInterval = minInterval;
        this.maxInterval = maxInterval;
        this.alphaBase = alphaBase;
        this.beta = beta;
        this.outlierThreshold = outlierThreshold;
        this.driftTolerance = driftTolerance;
        this.varianceSensitivity = varianceSensitivity;
        this.sigmaRef = sigmaRef;
        this.warmup = warmup;
        this.sigmaRefAdapt = sigmaRefAdapt;
        this.instabilityWindow = instabilityWindow;
        this.instabilityWeight = instabilityWeight;
        this.driftBoost = driftBoost;
        this.driftDecay = driftDecay;
        this.cooldownRate = cooldownRate;
        this.urgencySmoothing = urgencySmoothing;
        this.cooldownThreshold = cooldownThreshold;
        this.outlierDecay = outlierDecay;
    }

    public static AgentConfig parse(String args) {
        String output = "omniflow-original-dacapo.csv";
        String policy = "omni";
        boolean writeRows = false;
        boolean verbose = false;
        boolean disableOriginalSampler = true;
        boolean skipH2Reset = false;
        boolean skipCassandraNativeCheck = false;
        boolean telemetryConsole = false;
        boolean exactBenchmarkSeconds = false;
        long finalMarkerSecond = -1L;
        String cycleMarkersFile = "";
        long cycleLengthMillis = 180000L;
        double uniformRate = 0.5;
        long seed = 1L;
        String omniSignal = "heap_raw";

        int minInterval = 1;
        int maxInterval = 20;
        double alphaBase = 0.05;
        double beta = 0.5;
        double outlierThreshold = 1.0;
        int driftTolerance = 3;
        double varianceSensitivity = 1.0;
        double sigmaRef = 0.0;
        int warmup = 5;
        double sigmaRefAdapt = 0.01;
        int instabilityWindow = 20;
        double instabilityWeight = 1.0;
        double driftBoost = 1.0;
        double driftDecay = 0.90;
        double cooldownRate = 0.10;
        double urgencySmoothing = 0.60;
        double cooldownThreshold = 0.20;
        double outlierDecay = 0.85;

        if (args != null && !args.trim().isEmpty()) {
            String[] pairs = args.split(",");
            for (String pair : pairs) {
                String[] kv = pair.split("=", 2);
                if (kv.length != 2) {
                    continue;
                }
                String key = kv[0].trim().toLowerCase(Locale.ROOT);
                String value = kv[1].trim();
                if ("output".equals(key)) output = value;
                else if ("policy".equals(key)) policy = value.toLowerCase(Locale.ROOT);
                else if ("write_rows".equals(key)) writeRows = Boolean.parseBoolean(value);
                else if ("verbose".equals(key)) verbose = Boolean.parseBoolean(value);
                else if ("disable_original_sampler".equals(key)) disableOriginalSampler = Boolean.parseBoolean(value);
                else if ("skip_h2_reset".equals(key)) skipH2Reset = Boolean.parseBoolean(value);
                else if ("skip_cassandra_native_check".equals(key)) skipCassandraNativeCheck = Boolean.parseBoolean(value);
                else if ("telemetry_console".equals(key)) telemetryConsole = Boolean.parseBoolean(value);
                else if ("exact_benchmark_seconds".equals(key) || "h2_exact_seconds".equals(key)) exactBenchmarkSeconds = Boolean.parseBoolean(value);
                else if ("final_marker_second".equals(key)) finalMarkerSecond = Long.parseLong(value);
                else if ("cycle_markers_file".equals(key)) cycleMarkersFile = value;
                else if ("cycle_length_millis".equals(key)) cycleLengthMillis = Long.parseLong(value);
                else if ("uniform_rate".equals(key)) uniformRate = Double.parseDouble(value);
                else if ("seed".equals(key)) seed = Long.parseLong(value);
                else if ("omni_signal".equals(key)) omniSignal = value.toLowerCase(Locale.ROOT);
                else if ("min_interval".equals(key)) minInterval = Integer.parseInt(value);
                else if ("max_interval".equals(key)) maxInterval = Integer.parseInt(value);
                else if ("alpha_base".equals(key)) alphaBase = Double.parseDouble(value);
                else if ("beta".equals(key)) beta = Double.parseDouble(value);
                else if ("outlier_threshold".equals(key)) outlierThreshold = Double.parseDouble(value);
                else if ("drift_tolerance".equals(key)) driftTolerance = Integer.parseInt(value);
                else if ("variance_sensitivity".equals(key)) varianceSensitivity = Double.parseDouble(value);
                else if ("sigma_ref".equals(key)) sigmaRef = Double.parseDouble(value);
                else if ("warmup".equals(key)) warmup = Integer.parseInt(value);
                else if ("sigma_ref_adapt".equals(key)) sigmaRefAdapt = Double.parseDouble(value);
                else if ("instability_window".equals(key)) instabilityWindow = Integer.parseInt(value);
                else if ("instability_weight".equals(key)) instabilityWeight = Double.parseDouble(value);
                else if ("drift_boost".equals(key)) driftBoost = Double.parseDouble(value);
                else if ("drift_decay".equals(key)) driftDecay = Double.parseDouble(value);
                else if ("cooldown_rate".equals(key)) cooldownRate = Double.parseDouble(value);
                else if ("urgency_smoothing".equals(key)) urgencySmoothing = Double.parseDouble(value);
                else if ("cooldown_threshold".equals(key)) cooldownThreshold = Double.parseDouble(value);
                else if ("outlier_decay".equals(key)) outlierDecay = Double.parseDouble(value);
                else throw new IllegalArgumentException("Unknown agent option: " + key);
            }
        }

        if (!("full".equals(policy) || "omni".equals(policy)
                || "uni".equals(policy) || "inv".equals(policy) || "adp".equals(policy))) {
            throw new IllegalArgumentException("policy must be full, omni, uni, inv, or adp, got: " + policy);
        }
        if (!("heap_raw".equals(omniSignal) || "heap_positive".equals(omniSignal)
                || "duration_nanos".equals(omniSignal))) {
            throw new IllegalArgumentException(
                    "omni_signal must be heap_raw, heap_positive, or duration_nanos");
        }
        if (uniformRate < 0.0 || uniformRate > 1.0 || Double.isNaN(uniformRate)) {
            throw new IllegalArgumentException("uniform_rate must be in [0,1]");
        }
        if (("inv".equals(policy) || "adp".equals(policy)) && uniformRate < 0.01) {
            throw new IllegalArgumentException("INV/ADP uniform_rate must be in [0.01,1]");
        }
        if (cycleLengthMillis <= 0L) {
            throw new IllegalArgumentException("cycle_length_millis must be > 0");
        }
        if (minInterval < 1) throw new IllegalArgumentException("min_interval must be >= 1");
        if (maxInterval < minInterval) throw new IllegalArgumentException("max_interval must be >= min_interval");
        if (alphaBase <= 0.0 || alphaBase >= 1.0) throw new IllegalArgumentException("alpha_base must be in (0,1)");
        if (beta < 0.0) throw new IllegalArgumentException("beta must be >= 0");
        if (outlierThreshold <= 0.0) throw new IllegalArgumentException("outlier_threshold must be > 0");
        if (driftTolerance < 1) throw new IllegalArgumentException("drift_tolerance must be >= 1");
        if (warmup < 1) throw new IllegalArgumentException("warmup must be >= 1");
        if (instabilityWindow < 1) throw new IllegalArgumentException("instability_window must be >= 1");
        requireUnit("sigma_ref_adapt", sigmaRefAdapt);
        requireUnit("drift_decay", driftDecay);
        requireUnit("cooldown_rate", cooldownRate);
        requireUnit("urgency_smoothing", urgencySmoothing);
        requireUnit("cooldown_threshold", cooldownThreshold);
        requireUnit("outlier_decay", outlierDecay);

        return new AgentConfig(
                output, policy, writeRows, verbose, disableOriginalSampler, skipH2Reset,
                skipCassandraNativeCheck, telemetryConsole, exactBenchmarkSeconds, finalMarkerSecond, cycleMarkersFile, cycleLengthMillis,
                uniformRate, seed, omniSignal,
                minInterval, maxInterval, alphaBase, beta, outlierThreshold,
                driftTolerance, varianceSensitivity, sigmaRef, warmup,
                sigmaRefAdapt, instabilityWindow, instabilityWeight, driftBoost,
                driftDecay, cooldownRate, urgencySmoothing, cooldownThreshold,
                outlierDecay);
    }

    private static void requireUnit(String name, double value) {
        if (value < 0.0 || value > 1.0 || Double.isNaN(value)) {
            throw new IllegalArgumentException(name + " must be in [0,1]");
        }
    }
}
