package handwritten_digits.neural_network.layers;

import handwritten_digits.math.VecF;
import handwritten_digits.neural_network.ActivationFunction;
import handwritten_digits.neural_network.layers.DenseLayer;

public class OutputLayer extends DenseLayer {

    public OutputLayer(int in, int neurons, ActivationFunction activation, WeightInit weightInit) {
        super(in, neurons, activation, weightInit);
    }

    public VecF processInput(VecF inputs) {
        return super.processInput(inputs);
    }
}
