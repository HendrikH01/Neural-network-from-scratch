package handwritten_digits.neural_network.gradient_updaters;

import handwritten_digits.math.MatrixF;
import handwritten_digits.math.VecF;
import handwritten_digits.neural_network.layers.DenseLayer;
import handwritten_digits.neural_network.layers.Layer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GradientDescent {

    public static Map<DenseLayer, VecF>  calculateError(List<Layer> layers, VecF nablaCost) {
        Map<DenseLayer, VecF> errors = new HashMap<>();

        //Last layer
        int index = layers.size() - 1;
        if(layers.get(index) instanceof DenseLayer) {
            DenseLayer outputLayer = (DenseLayer)layers.get(index);

            VecF error = nablaCost.mult(
                    outputLayer.activationFunction.derivative(outputLayer.zValueCache));

            errors.put(outputLayer, error);

        } else return errors;

        //Backpropagation
        DenseLayer prev = (DenseLayer)layers.get(index);
        VecF prevError = errors.get(prev);

        index--;

        while(index >= 0) {
            if(layers.get(index) instanceof DenseLayer) {
                DenseLayer denseLayer = (DenseLayer)layers.get(index);
                VecF error = calculateError(denseLayer, prev, prevError);
                errors.put(denseLayer, error);

                prevError = error;
                prev = denseLayer;
            } else {
                break;
            }

            index--;
        }

        return errors;
    }

    private static VecF calculateError(DenseLayer layer, DenseLayer previous, VecF prevError) {
        return previous.weights.transpose().mult(prevError)
                .mult(layer.activationFunction.derivative(layer.zValueCache));
    }

    /**
     * Calculates dC/dw by multiplying the error with the activation of the neuron of first layer the weight is connected to
     *
     * @param errors
     * @param layers
     * @param input
     * @return
     */
    public static Map<DenseLayer, MatrixF> weightGradient(Map<DenseLayer, VecF> errors, List<Layer> layers, VecF input) {
        Map<DenseLayer, MatrixF> grad = new HashMap<>();

        DenseLayer previous;

        if (layers.get(0) instanceof DenseLayer) {
            DenseLayer denseLayer = (DenseLayer) layers.get(0);
            calcWeightGradientAt(denseLayer, input, errors, grad);
            previous = denseLayer;

        } else return grad;

        for (int i = 1; i < layers.size(); i++) {

            if (layers.get(i) instanceof DenseLayer) {
                DenseLayer denseLayer = (DenseLayer) layers.get(i);
                calcWeightGradientAt(denseLayer, previous.activationCache, errors, grad);

                previous = denseLayer;
            }
        }

        return grad;
    }

    private static void calcWeightGradientAt(DenseLayer denseLayer, VecF prevActivation, Map<DenseLayer, VecF> errors, Map<DenseLayer, MatrixF> grad) {
        MatrixF matrix = new MatrixF(prevActivation.getLength(), denseLayer.outSize);
        VecF error = errors.get(denseLayer);

        for(int i = 0; i < matrix.getWidth(); i++) {
            for(int j = 0; j < matrix.getHeight(); j++) {
                matrix.set(i, j, error.get(j) * prevActivation.get(i));
            }
        }

        grad.put(denseLayer, matrix);
    }

    /**
     * Calculates dC/db, which is very easy since that's just the error at the current neuron
     *
     * @param errors
     * @param layers
     * @return
     */
    public static Map<DenseLayer, VecF> biasGradient(Map<DenseLayer, VecF> errors, List<Layer> layers) {
        return new HashMap<>(errors);
    }
}
