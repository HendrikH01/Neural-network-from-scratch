package handwritten_digits.neural_network.gradient_updaters;

import handwritten_digits.math.MatrixF;
import handwritten_digits.math.VecF;
import handwritten_digits.neural_network.NetworkModel;
import handwritten_digits.neural_network.layers.DenseLayer;
import handwritten_digits.neural_network.layers.Layer;

import java.util.List;
import java.util.Map;

public class AdamGradientUpdater extends GradientUpdater<AdamConfig> {

    private static final AdamGradientUpdater INSTANCE = new AdamGradientUpdater();

    private float[] prevMoment = null;
    private float[] prevVariance = null;

    public AdamGradientUpdater(AdamConfig config) {
        super(config);
    }

    public AdamGradientUpdater() {
        super(AdamConfig.DEFAULT_INSTANCE);
    }

    @Override
    public void init(int networkParameterCount) {
        this.prevMoment = new float[networkParameterCount];
        this.prevVariance = new float[networkParameterCount];
    }

    @Override
    public void applyUpdater(float[] parameters, float[] gradient) {
        for(int i = 0; i < parameters.length; i++) {
            prevMoment[i] = (prevMoment[i] * config.beta1 + (1.0F - config.beta1) * gradient[i]);
            prevVariance[i] = (prevVariance[i] * config.beta2 + (1.0F - config.beta2) * gradient[i] * gradient[i]);
            float mHat = prevMoment[i] / (1.0F - config.beta1);
            float vHat = prevVariance[i] / (1.0F - config.beta2);
            parameters[i] -= config.alpha * mHat / (Math.sqrt(vHat) + config.epsilon);
        }
    }
}
