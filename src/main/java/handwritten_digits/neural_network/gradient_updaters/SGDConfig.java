package handwritten_digits.neural_network.gradient_updaters;

public class SGDConfig implements IGradientUpdaterConfig {

    public final float learningRate;
    public final int miniBatchSize;

    public SGDConfig(float learningRate, int miniBatchSize) {
        this.learningRate = learningRate;
        this.miniBatchSize = miniBatchSize;
    }
}
