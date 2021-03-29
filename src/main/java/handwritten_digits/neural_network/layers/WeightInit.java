package handwritten_digits.neural_network.layers;

import handwritten_digits.math.VecF;

import java.util.Random;

public enum WeightInit {
    UNIFORM,
    XAVIER,
    GAUSSIAN,
    ZEROS,
    ONES;

    public float next(Random random, int in) {
        switch (this) {
            case UNIFORM:
                return random.nextFloat() - 0.5F;
            case XAVIER:
                return (float) ((random.nextFloat() - 0.5F) * 2.0F / Math.sqrt(in));
            case GAUSSIAN:
                return (float) random.nextGaussian();
            case ZEROS:
                return 0.0F;
            case ONES:
                return 1.0F;
        }

        return 0;
    }
}
