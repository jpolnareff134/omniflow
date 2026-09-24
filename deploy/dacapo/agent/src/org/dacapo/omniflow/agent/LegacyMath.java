package org.dacapo.omniflow.agent;

/** Small numerical subset needed to reproduce Commons Math's one-sample t test. */
final class LegacyMath {
    private LegacyMath() { }

    static boolean oneSampleTTestReject(double mu, RunningStats statistics, double alpha) {
        long n = statistics.count();
        if (n < 2L || !(alpha > 0.0 && alpha <= 0.5)) return false;
        double variance = statistics.variance();
        if (!(variance > 0.0) || Double.isNaN(variance)) return false;
        double t = (statistics.mean() - mu) / Math.sqrt(variance / (double) n);
        double x = (double) (n - 1L) / ((double) (n - 1L) + t * t);
        double pTwoSided = regularizedBeta(x, 0.5 * (double) (n - 1L), 0.5);
        return pTwoSided < alpha;
    }

    private static double regularizedBeta(double x, double a, double b) {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        double logTerm = logGamma(a + b) - logGamma(a) - logGamma(b)
                + a * Math.log(x) + b * Math.log1p(-x);
        double bt = Math.exp(logTerm);
        if (x < (a + 1.0) / (a + b + 2.0)) {
            return bt * betaFraction(a, b, x) / a;
        }
        return 1.0 - bt * betaFraction(b, a, 1.0 - x) / b;
    }

    private static double betaFraction(double a, double b, double x) {
        final int maxIterations = 10000;
        final double epsilon = 3.0e-14;
        final double fpMin = 1.0e-300;
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        if (Math.abs(d) < fpMin) d = fpMin;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= maxIterations; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < fpMin) d = fpMin;
            c = 1.0 + aa / c;
            if (Math.abs(c) < fpMin) c = fpMin;
            d = 1.0 / d;
            h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < fpMin) d = fpMin;
            c = 1.0 + aa / c;
            if (Math.abs(c) < fpMin) c = fpMin;
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (Math.abs(delta - 1.0) < epsilon) break;
        }
        return h;
    }

    private static double logGamma(double x) {
        double[] coefficients = {
                676.5203681218851, -1259.1392167224028,
                771.32342877765313, -176.61502916214059,
                12.507343278686905, -0.13857109526572012,
                9.9843695780195716e-6, 1.5056327351493116e-7
        };
        if (x < 0.5) {
            return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * x)) - logGamma(1.0 - x);
        }
        double y = x - 1.0;
        double a = 0.99999999999980993;
        for (int i = 0; i < coefficients.length; i++) {
            a += coefficients[i] / (y + i + 1.0);
        }
        double t = y + coefficients.length - 0.5;
        return 0.5 * Math.log(2.0 * Math.PI) + (y + 0.5) * Math.log(t) - t + Math.log(a);
    }
}
