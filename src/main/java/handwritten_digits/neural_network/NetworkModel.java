package handwritten_digits.neural_network;

import handwritten_digits.analysis.IEvaluator;
import handwritten_digits.math.MatrixF;
import handwritten_digits.math.VecF;
import handwritten_digits.neural_network.gradient_updaters.GradientDescent;
import handwritten_digits.neural_network.gradient_updaters.GradientUpdater;
import handwritten_digits.neural_network.gradient_updaters.IGradientUpdaterConfig;
import handwritten_digits.neural_network.layers.DenseLayer;
import handwritten_digits.neural_network.layers.Layer;
import handwritten_digits.util.DataSet;
import handwritten_digits.util.Example;

import java.util.*;

public class NetworkModel {

    protected Random rand;
    protected List<Layer> layers = new ArrayList<>();
    protected CostFunction cost;
    protected GradientUpdater<? extends IGradientUpdaterConfig> gradientUpdater;
    protected int parameterCount = 0;

    public NetworkModel() {}

    /**
     * Initializes the weights and biases with random floats between 0 and 1
     */
    public void init() {
        for(Layer layer : this.layers) {
            layer.init(this.rand);
            this.parameterCount += layer.getParameterCount();
        }

        this.gradientUpdater.init(this.parameterCount);
    }

    /**
     * Trains the neural network
     *
     * @param batchSize
     * @param epochs
     * @param data
     */
    public void train(int batchSize, int epochs, DataSet data) {

        for(int epoch = 0; epoch < epochs; epoch++) {
            int examples = 0;
            double averageCost = 0;

            while(data.hasNextBatch()) {
                DataSet batch = data.nextBatch(batchSize);

                while(batch.hasNext()) {
                    Example<VecF, VecF> example = batch.next();
                    examples++;

                    VecF output = this.propagateData(new VecF(example.feature), 0);

                    if(this.gradientUpdater.shouldUpdate(epoch, examples)) {

                        this.updateParameters(example.feature, output, example.label);
                    }

                    averageCost += this.determineCost(output, example.label);
                }
            }

            data.reset();

            System.out.printf("Average cost in epoch %d: %.2f\n", epoch, averageCost / (double) examples);
        }
    }

    /**
     * Recursively feeds data through neural network
     *
     * @param data
     * @param layerIndex
     * @return network output
     */
    private VecF propagateData(VecF data, int layerIndex) {
        if(layerIndex >= this.layers.size()) {
            return data;
        } else {
            return propagateData(this.layers.get(layerIndex).processInput(data), ++layerIndex);
        }
    }

    private float determineCost(VecF networkOutput, VecF expected) {
        return this.cost.apply(networkOutput, expected);
    }

    /**
     * Uses gradient descent and backpropagation to update the weights and biases in this network.
     */
    private void updateParameters(VecF input, VecF output, VecF expected) {
        VecF nablaCost = this.cost.derivative(output, expected);

        Map<DenseLayer, VecF> error = GradientDescent.calculateError(this.layers, nablaCost);
        Map<DenseLayer, VecF> biasGradient = GradientDescent.biasGradient(error, this.layers);
        Map<DenseLayer, MatrixF> weightGradient = GradientDescent.weightGradient(error, this.layers, input);

        //Turn parameters into float array
        float[] parameters = new float[this.parameterCount];
        float[] gradient = new float[this.parameterCount];
        int index = 0;

        for(Layer l : this.layers) {
            if (l.getParameterCount() > 0) {

                if(biasGradient.containsKey(l) && weightGradient.containsKey(l)) {
                    System.arraycopy(l.getParameters(), 0, parameters, index, l.getParameterCount());
                    float[] weightGrad = weightGradient.get(l).toArray();
                    System.arraycopy(weightGrad, 0, gradient, index, weightGrad.length);
                    System.arraycopy(biasGradient.get(l).toArray(), 0, gradient, index + weightGrad.length, biasGradient.get(l).getLength());

                    index += l.getParameterCount();

                }
            }
        }

        this.gradientUpdater.applyUpdater(parameters, gradient);

        //Turn float array back into parameters
        index = 0;
        for(Layer l : this.layers) {
            if (l.getParameterCount() > 0) {
                float[] params = new float[l.getParameterCount()];
                System.arraycopy(parameters, index, params, 0, l.getParameterCount());

                l.setParameters(params);
                index += l.getParameterCount();

            }
        }


    }

    /**
     * Tests the accuracy of the network
     *
     * @param testingData
     * @return accuracy of the network
     */
    public void test(int batchSize, DataSet testingData, IEvaluator eval) {

        while(testingData.hasNextBatch()) {
            DataSet batch = testingData.nextBatch(batchSize);

            while(batch.hasNext()) {
                Example<VecF, VecF> example = batch.next();

                VecF output = this.propagateData(new VecF(example.feature), 0);
                eval.compareResult(output, example.label);
            }
        }

        eval.printStats();
    }

    public static class Builder extends NetworkModel {

        public Builder setRandomSeed(long seed) {
            this.rand = new Random(seed);
            return this;
        }

        public Builder setCostFunction(CostFunction cost) {
            this.cost = cost;
            return this;
        }

        public Builder setGradientUpdater(GradientUpdater gradientUpdater) {
            this.gradientUpdater = gradientUpdater;
            return this;
        }

        public NetworkModel build() {
            return this;
        }

        public Builder addLayer(Layer layer) {
            this.layers.add(layer);
            return this;
        }
    }
}
