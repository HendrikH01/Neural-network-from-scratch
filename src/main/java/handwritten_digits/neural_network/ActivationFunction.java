package handwritten_digits.neural_network;

import handwritten_digits.math.VecF;
import handwritten_digits.util.Util;

import java.util.function.Function;

public enum ActivationFunction {
    RELU,
    SIGMOID,
    SOFTMAX;

    public VecF apply(VecF vec) {
        switch (this) {
            case RELU:
                return relu(vec);
            case SIGMOID:
                return sigmoid(vec);
            case SOFTMAX:
                return softmax(vec);
            default:
                throw new IllegalArgumentException();
        }
    }

    public VecF derivative(VecF vec) {
        switch (this) {
            case RELU:
                return dRelu(vec);
            case SIGMOID:
                return dSigmoid(vec);
            case SOFTMAX:
                return dSoftmax(vec);
            default:
                throw new IllegalArgumentException();
        }
    }

    private static VecF relu(VecF vec) {
        VecF out = new VecF(vec.getLength());

        for(int i = 0; i < vec.getLength(); i++) {
            out.set(i, vec.get(i) > 0 ? vec.get(i) : 0);
        }

        return out;
    }

    private static VecF dRelu(VecF f) {
        VecF out = new VecF(f.getLength());

        for(int i = 0; i < f.getLength(); i++) {
            out.set(i, f.get(i) > 0 ? 1 : 0);
        }

        return out;
    }

    private static VecF sigmoid(VecF f) {
        VecF out = new VecF(f.getLength());

        for(int i = 0; i < f.getLength(); i++) {
            out.set(i, (float) (1/(1 + Math.pow(Math.E, -f.get(i)))));
        }

        return out;
    }

    private static VecF dSigmoid(VecF f) {
        VecF out = sigmoid(f);

        //return out_i * (1 - out_i)
        return out.mult(new VecF(f.getLength(), 1.0F).sub(out));
    }

    private static VecF softmax(VecF f) {
        VecF out = new VecF(f.getLength());
        float max = Util.getLargestElement(f);
        double divisor = 0;

        for(int i = 0; i < f.getLength(); i++) {
            out.set(i, (float) Math.pow(Math.E, f.get(i) - max));
            divisor += out.get(i);
        }

        out = out.div(divisor);

        return out;
    }

    private static VecF dSoftmax(VecF f) {
        VecF out = softmax(f);

        //return out_i * (1 - out_i)
        return out.mult(new VecF(f.getLength(), 1.0F).sub(out));
    }
}
