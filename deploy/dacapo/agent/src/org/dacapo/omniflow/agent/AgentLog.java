package org.dacapo.omniflow.agent;

import java.io.PrintStream;

/**
 * Agent diagnostics that bypass DaCapo's temporary System.err capture.
 *
 * DaCapo validates benchmark-owned stdout/stderr files.  The agent captures
 * the original stream during premain and keeps using that object even if the
 * harness later calls System.setErr(...).  This preserves diagnostics in the
 * outer run log without contaminating DaCapo's digest inputs.
 */
public final class AgentLog {
    private static volatile PrintStream originalErr = System.err;

    private AgentLog() { }

    public static void captureOriginalStreams() {
        originalErr = System.err;
    }

    public static void info(String message) {
        PrintStream stream = originalErr;
        synchronized (stream) {
            stream.println("[OmniFlowAgent] " + message);
            stream.flush();
        }
    }

    public static void error(String message, Throwable error, boolean verbose) {
        PrintStream stream = originalErr;
        synchronized (stream) {
            stream.println("[OmniFlowAgent] " + message);
            if (verbose && error != null) {
                error.printStackTrace(stream);
            }
            stream.flush();
        }
    }
}
