package org.dacapo.omniflow.agent;

import java.lang.instrument.Instrumentation;

/** Entry point for the Java 8-compatible DaCapo agent. */
public final class OmniFlowAgent {
    private OmniFlowAgent() { }

    public static void premain(String agentArgs, Instrumentation instrumentation) {
        AgentLog.captureOriginalStreams();
        final AgentConfig config = AgentConfig.parse(agentArgs);
        RequestTracker.initialize(config);
        instrumentation.addTransformer(new OmniFlowTransformer(config));
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                RequestTracker.shutdown();
            }
        }, "omniflow-agent-shutdown"));
    }

    public static void agentmain(String agentArgs, Instrumentation instrumentation) {
        premain(agentArgs, instrumentation);
    }
}
