package handwritten_digits.neural_network.gradient_updaters;

import handwritten_digits.math.MatrixF;
import handwritten_digits.math.VecF;
import handwritten_digits.neural_network.layers.DenseLayer;
import handwritten_digits.neural_network.layers.Layer;

import java.util.List;
import java.util.Map;

/**
 * SGD mini-batch implementation
 */
public class SGD extends GradientUpdater<SGDConfig> {

    public SGD(SGDConfig config) {
        super(config);
    }

    @Override
    public void applyUpdater(float[] parameters, float[] gradient) {

        for(int i = 0; i < gradient.length; i++) {
            parameters[i] -= gradient[i] * this.config.learningRate;
        }
    }

    @Override
    public boolean shouldUpdate(int epoch, int example) {
        return example % this.config.miniBatchSize == 0;
    }
}
