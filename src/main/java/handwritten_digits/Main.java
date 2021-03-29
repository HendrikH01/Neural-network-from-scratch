package handwritten_digits;

import handwritten_digits.analysis.ClassificationEvaluator;
import handwritten_digits.math.VecF;
import handwritten_digits.neural_network.ActivationFunction;
import handwritten_digits.neural_network.CostFunction;
import handwritten_digits.neural_network.gradient_updaters.AdamGradientUpdater;
import handwritten_digits.neural_network.layers.DenseLayer;
import handwritten_digits.neural_network.NetworkModel;
import handwritten_digits.neural_network.layers.OutputLayer;
import handwritten_digits.util.DataSet;
import handwritten_digits.util.MNISTLoader;

import java.io.IOException;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws IOException {

        int batchSize = 64; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 5; // number of epochs to perform

        DataSet data = MNISTLoader.loadMNIST(true, rngSeed);//binaryORTraining(1, 5000, rngSeed);
        DataSet test = MNISTLoader.loadMNIST(false, rngSeed);//binaryORTraining(1, 5000, rngSeed);

        NetworkModel networkModel = new NetworkModel.Builder().setRandomSeed(rngSeed).setCostFunction(CostFunction.CROSS_ENTROPY)
                .setGradientUpdater(new AdamGradientUpdater())
                .addLayer(new DenseLayer(784, 40, ActivationFunction.SIGMOID, handwritten_digits.neural_network.layers.WeightInit.XAVIER))
                .addLayer(new DenseLayer(40, 30, ActivationFunction.SIGMOID, handwritten_digits.neural_network.layers.WeightInit.XAVIER))
                .addLayer(new OutputLayer(30, 10, ActivationFunction.SOFTMAX, handwritten_digits.neural_network.layers.WeightInit.XAVIER))
                .build();

        networkModel.init();

        networkModel.train(batchSize, numEpochs, data);

        networkModel.test(batchSize, test, new ClassificationEvaluator(10));

    }

    /**
     *
     * @param length
     * @param rngSeed
     * @return
     */
    public static DataSet binaryORTraining(int size, int length, long rngSeed) {
        VecF[] features = new VecF[length];
        VecF[] labels = new VecF[length];
        Random r = new Random(rngSeed);
        for(int i = 0; i < length; i++) {
            int a = r.nextInt(2 << size);
            int b = r.nextInt(2 << size);

            features[i] = new VecF((float) a, (float) b);
            labels[i] = new VecF((float) (a | b) / (2 << size));
        }

        return new DataSet(features, labels);
    }
}
