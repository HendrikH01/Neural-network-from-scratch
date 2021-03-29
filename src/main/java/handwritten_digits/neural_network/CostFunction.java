package handwritten_digits.neural_network;

import handwritten_digits.math.VecF;

public enum CostFunction {
    QUADRATIC,
    CROSS_ENTROPY,
    KULLBACK_LEIBLER;

    public float apply(VecF output, VecF expected) {
        switch (this) {
            case QUADRATIC:
                return quadratic(output, expected);
            case CROSS_ENTROPY:
                return crossEntropy(output, expected);
            case KULLBACK_LEIBLER:
                return kullbackLeibler(output, expected);
            default:
                throw new IllegalArgumentException();
        }
    }

    public VecF derivative(VecF output, VecF expected) {
        switch (this) {
            case QUADRATIC:
                return nablaQuadratic(output, expected);
            case CROSS_ENTROPY:
                return nablaCrossEntropy(output, expected);
            case KULLBACK_LEIBLER:
                return nablaKullbackLeibler(output, expected);
            default:
                throw new IllegalArgumentException();
        }
    }

    private float quadratic(VecF output, VecF expected) {
        double sum = 0;
        for(int i = 0; i < output.getLength(); i++) {
            sum += Math.pow(expected.get(i) - output.get(i), 2);
        }

        return (float) sum / 2;
    }

    private VecF nablaQuadratic(VecF output, VecF expected) {
        return output.sub(expected);
    }

    private float kullbackLeibler(VecF output, VecF expected) {
        double sum = 0;
        for(int i = 0; i < output.getLength(); i++) {
            sum += expected.get(i) * Math.log(expected.get(i) / output.get(i));
        }

        return (float) sum / 2;
    }

    private VecF nablaKullbackLeibler(VecF output, VecF expected) {
        return expected.div(output).mult(-1);
    }

    private float crossEntropy(VecF output, VecF expected) {
        double sum = 0;
        for(int i = 0; i < output.getLength(); i++) {
            sum -= expected.get(i) * Math.log(output.get(i)) + (1 - expected.get(i)) * Math.log(1 - output.get(i));
        }

        return (float) sum;
    }

    private VecF nablaCrossEntropy(VecF output, VecF expected) {
        return output.sub(expected).div(output.sub(output.mult(output)).clamp(0.000001F, 100000.0F));
    }
}
