package handwritten_digits.neural_network.gradient_updaters;

import handwritten_digits.math.MatrixF;
import handwritten_digits.math.VecF;
import handwritten_digits.neural_network.NetworkModel;
import handwritten_digits.neural_network.layers.DenseLayer;
import handwritten_digits.neural_network.layers.Layer;

import java.util.List;
import java.util.Map;

public abstract class GradientUpdater<T extends IGradientUpdaterConfig> {

    protected final T config;

    public GradientUpdater(T config) {
        this.config = config;
    }

    public void init(int networkParameterCount) {};

    public abstract void applyUpdater(float[] parameters, float[] gradient);

    public boolean shouldUpdate(int epoch, int example) {
        return true;
    }
}
