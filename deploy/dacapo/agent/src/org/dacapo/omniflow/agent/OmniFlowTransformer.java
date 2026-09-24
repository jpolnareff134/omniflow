package org.dacapo.omniflow.agent;

import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Label;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.Type;
import org.objectweb.asm.commons.AdviceAdapter;
import org.objectweb.asm.commons.Method;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Instruments the exact request entry points used by the original artifact. */
public final class OmniFlowTransformer implements ClassFileTransformer {
    private static final String TRACKER = "org/dacapo/omniflow/agent/RequestTracker";
    private static final String ORIGINAL_ASPECT = "br/ufrgs/inf/prosoft/tigris/sampling/SamplingAspect";
    private static final String H2_TPCC = "org/dacapo/h2/TPCC";
    private static final String H2_REPORTER = "org/dacapo/h2/TPCCReporter";
    private static final String CASSANDRA_NATIVE_CHECK = "org/apache/cassandra/service/StartupChecks$6";
    private static final String CASSANDRA_NATIVE_LIBRARY = "org/apache/cassandra/utils/NativeLibrary";
    private final AgentConfig config;

    public OmniFlowTransformer(AgentConfig config) {
        this.config = config;
    }

    @Override
    public byte[] transform(ClassLoader loader, String className, Class<?> classBeingRedefined,
                            ProtectionDomain protectionDomain, byte[] classfileBuffer) {
        if (className == null) return null;
        boolean disableAspect = config.disableOriginalSampler && ORIGINAL_ASPECT.equals(className);
        boolean instrumentExactBenchmarkSeconds = config.exactBenchmarkSeconds && ORIGINAL_ASPECT.equals(className);
        boolean skipH2Reset = config.skipH2Reset && H2_TPCC.equals(className);
        boolean skipCassandraNativeCheck = config.skipCassandraNativeCheck && CASSANDRA_NATIVE_CHECK.equals(className);
        boolean useCassandraPidFallback = config.skipCassandraNativeCheck && CASSANDRA_NATIVE_LIBRARY.equals(className);
        boolean instrumentH2Throughput = !config.exactBenchmarkSeconds && ("inv".equals(config.policy)
                || "adp".equals(config.policy)
                || (config.cycleMarkersFile != null && !config.cycleMarkersFile.trim().isEmpty()))
                && H2_REPORTER.equals(className);
        List<Target> targets = Target.forClass(className);
        if (!disableAspect && !instrumentExactBenchmarkSeconds && !skipH2Reset && !skipCassandraNativeCheck
                && !useCassandraPidFallback && !instrumentH2Throughput && (targets == null || targets.isEmpty())) return null;

        try {
            ClassReader reader = new ClassReader(classfileBuffer);
            ClassWriter writer = new ClassWriter(reader, ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
            ClassVisitor visitor = new Visitor(writer, targets, disableAspect,
                    instrumentExactBenchmarkSeconds, skipH2Reset, skipCassandraNativeCheck, useCassandraPidFallback, instrumentH2Throughput,
                    config.verbose, className);
            reader.accept(visitor, ClassReader.EXPAND_FRAMES);
            return writer.toByteArray();
        } catch (Throwable error) {
            AgentLog.error("transform failed for " + className + ": " + error, error, config.verbose);
            return null;
        }
    }

    private static final class Target {
        final String className;
        final String methodName;
        final String descriptor;
        final String methodId;
        final int argumentIndex;

        Target(String className, String methodName, String descriptor,
               String methodId, int argumentIndex) {
            this.className = className;
            this.methodName = methodName;
            this.descriptor = descriptor;
            this.methodId = methodId;
            this.argumentIndex = argumentIndex;
        }

        boolean matches(String name, String desc) {
            return methodName.equals(name) && (descriptor == null || descriptor.equals(desc));
        }

        private static final Map<String, List<Target>> ALL = new HashMap<String, List<Target>>();
        static {
            add("org/dacapo/h2/TPCCSubmitter", "runTransaction", "(ILjava/lang/Object;)Z", 0);
            add("org/dacapo/lusearch/QueryProcessor", "doPagingSearch", "(Lorg/apache/lucene/search/Query;)V", 0);
            add("org/dacapo/xalan/XalanWorker", "transform", "(Ljavax/xml/transform/Result;Ljava/lang/String;)V", 1);

            String trader = "org/apache/geronimo/daytrader/javaee6/dacapo/DaCapoTrader";
            add(trader, "doHome", null, 0);
            add(trader, "doPortfolio", null, 0);
            add(trader, "doQuote", null, 0);
            add(trader, "doBuy", null, 0);
            add(trader, "doUpdate", null, 0);
            add(trader, "doRegister", null, 0);
            add(trader, "doSell", null, 0);

            String ycsb = "site/ycsb/workloads/CoreWorkload";
            add(ycsb, "doTransactionRead", null, 0);
            add(ycsb, "doTransactionUpdate", null, 0);
            add(ycsb, "doTransactionInsert", null, 0);
            add(ycsb, "doTransactionScan", null, 0);
            add(ycsb, "doTransactionReadModifyWrite", null, 0);
        }

        private static void add(String className, String methodName, String descriptor, int argumentIndex) {
            List<Target> list = ALL.get(className);
            if (list == null) {
                list = new ArrayList<Target>();
                ALL.put(className, list);
            }
            String methodId = className.replace('/', '.') + "." + methodName;
            list.add(new Target(className, methodName, descriptor, methodId, argumentIndex));
        }

        static List<Target> forClass(String className) {
            return ALL.get(className);
        }
    }

    private static final class Visitor extends ClassVisitor {
        private final List<Target> targets;
        private final boolean disableAspect;
        private final boolean instrumentExactBenchmarkSeconds;
        private final boolean skipH2Reset;
        private final boolean skipCassandraNativeCheck;
        private final boolean useCassandraPidFallback;
        private final boolean instrumentH2Throughput;
        private final boolean verbose;
        private final String className;

        Visitor(ClassVisitor delegate, List<Target> targets, boolean disableAspect,
                boolean instrumentExactBenchmarkSeconds, boolean skipH2Reset, boolean skipCassandraNativeCheck, boolean useCassandraPidFallback, boolean instrumentH2Throughput,
                boolean verbose, String className) {
            super(Opcodes.ASM5, delegate);
            this.targets = targets;
            this.disableAspect = disableAspect;
            this.instrumentExactBenchmarkSeconds = instrumentExactBenchmarkSeconds;
            this.skipH2Reset = skipH2Reset;
            this.skipCassandraNativeCheck = skipCassandraNativeCheck;
            this.useCassandraPidFallback = useCassandraPidFallback;
            this.instrumentH2Throughput = instrumentH2Throughput;
            this.verbose = verbose;
            this.className = className;
        }

        @Override
        public MethodVisitor visitMethod(int access, String name, String descriptor,
                                         String signature, String[] exceptions) {
            MethodVisitor base = super.visitMethod(access, name, descriptor, signature, exceptions);
            if (disableAspect && "<clinit>".equals(name)) {
                return new DisableAspectAdapter(base, access, name, descriptor);
            }
            if (instrumentExactBenchmarkSeconds && "addOperationsPerSecondAndAdapt".equals(name)
                    && "(IILjava/util/Set;)V".equals(descriptor)) {
                AgentLog.info("exact benchmark per-second hook enabled");
                return new ExactBenchmarkSecondsAdapter(base, access, name, descriptor);
            }
            if (skipH2Reset && "resetToInitialData".equals(name) && "()V".equals(descriptor)) {
                AgentLog.info("H2 post-iteration reset disabled");
                return new SkipH2ResetAdapter(base, access, name, descriptor);
            }
            if (skipCassandraNativeCheck && "execute".equals(name) && "()V".equals(descriptor)) {
                AgentLog.info("Cassandra native-library startup check bypass enabled (development only)");
                return new SkipVoidMethodAdapter(base, access, name, descriptor);
            }
            if (useCassandraPidFallback && "getProcessID".equals(name) && "()J".equals(descriptor)) {
                AgentLog.info("Cassandra native PID fallback enabled (development only)");
                return new ReturnMinusOneLongAdapter(base, access, name, descriptor);
            }
            if (instrumentH2Throughput && "done".equals(name) && "()V".equals(descriptor)) {
                AgentLog.info("H2 top-level throughput hook enabled");
                return new H2ThroughputAdapter(base, access, name, descriptor);
            }
            if (targets != null) {
                for (Target target : targets) {
                    if (target.matches(name, descriptor)) {
                        if (verbose) {
                            AgentLog.info("instrument " + className + "." + name + descriptor);
                        }
                        return new RequestAdapter(base, access, name, descriptor, target);
                    }
                }
            }
            return base;
        }
    }

    private static final class DisableAspectAdapter extends AdviceAdapter {
        DisableAspectAdapter(MethodVisitor delegate, int access, String name, String descriptor) {
            super(Opcodes.ASM5, delegate, access, name, descriptor);
        }

        @Override
        protected void onMethodExit(int opcode) {
            if (opcode == RETURN) {
                push(false);
                putStatic(Type.getObjectType(ORIGINAL_ASPECT), "enabled", Type.BOOLEAN_TYPE);
            }
        }
    }



    private static final class ExactBenchmarkSecondsAdapter extends AdviceAdapter {
        ExactBenchmarkSecondsAdapter(MethodVisitor delegate, int access, String name, String descriptor) {
            super(Opcodes.ASM5, delegate, access, name, descriptor);
        }

        @Override
        protected void onMethodEnter() {
            loadArg(1);
            loadArg(0);
            invokeStatic(Type.getObjectType(TRACKER),
                    new Method("operationsPerSecond", "(II)V"));
        }
    }

    private static final class SkipH2ResetAdapter extends AdviceAdapter {
        SkipH2ResetAdapter(MethodVisitor delegate, int access, String name, String descriptor) {
            super(Opcodes.ASM5, delegate, access, name, descriptor);
        }

        @Override
        protected void onMethodEnter() {
            visitInsn(RETURN);
        }
    }


    private static final class SkipVoidMethodAdapter extends AdviceAdapter {
        SkipVoidMethodAdapter(MethodVisitor delegate, int access, String name, String descriptor) {
            super(Opcodes.ASM5, delegate, access, name, descriptor);
        }

        @Override
        protected void onMethodEnter() {
            visitInsn(RETURN);
        }
    }

    private static final class ReturnMinusOneLongAdapter extends AdviceAdapter {
        ReturnMinusOneLongAdapter(MethodVisitor delegate, int access, String name, String descriptor) {
            super(Opcodes.ASM5, delegate, access, name, descriptor);
        }

        @Override
        protected void onMethodEnter() {
            push(-1L);
            visitInsn(LRETURN);
        }
    }

    private static final class H2ThroughputAdapter extends AdviceAdapter {
        H2ThroughputAdapter(MethodVisitor delegate, int access, String name, String descriptor) {
            super(Opcodes.ASM5, delegate, access, name, descriptor);
        }

        @Override
        protected void onMethodExit(int opcode) {
            if (opcode == RETURN) {
                invokeStatic(Type.getObjectType(TRACKER),
                        new Method("operationCompleted", "()V"));
            }
        }
    }

    private static final class RequestAdapter extends AdviceAdapter {
        private final Target target;
        private final Type[] argumentTypes;
        private final Label protectedStart = new Label();

        RequestAdapter(MethodVisitor delegate, int access, String name, String descriptor, Target target) {
            super(Opcodes.ASM5, delegate, access, name, descriptor);
            this.target = target;
            this.argumentTypes = Type.getArgumentTypes(descriptor);
        }

        @Override
        protected void onMethodEnter() {
            push(target.methodId);
            loadArg(target.argumentIndex);
            box(argumentTypes[target.argumentIndex]);
            invokeStatic(Type.getObjectType(TRACKER),
                    new Method("requestType", "(Ljava/lang/String;Ljava/lang/Object;)Ljava/lang/String;"));
            invokeStatic(Type.getObjectType(TRACKER),
                    new Method("startRequest", "(Ljava/lang/String;)V"));
            mark(protectedStart);
        }

        @Override
        protected void onMethodExit(int opcode) {
            // ATHROW only covers explicit throw instructions. A catch-all handler
            // added in visitMaxs handles both explicit and implicit exceptions,
            // ensuring ThreadLocal depth cannot become permanently unbalanced.
            if (opcode != ATHROW) {
                invokeStatic(Type.getObjectType(TRACKER), new Method("endRequest", "()V"));
            }
        }

        @Override
        public void visitMaxs(int maxStack, int maxLocals) {
            Label protectedEnd = new Label();
            mark(protectedEnd);
            catchException(protectedStart, protectedEnd, Type.getType(Throwable.class));
            invokeStatic(Type.getObjectType(TRACKER), new Method("endRequest", "()V"));
            throwException();
            super.visitMaxs(maxStack, maxLocals);
        }
    }
}
