package handwritten_digits.neural_network.layers;

import handwritten_digits.math.MatrixF;
import handwritten_digits.math.VecF;
import handwritten_digits.neural_network.ActivationFunction;

import java.util.Arrays;
import java.util.Random;

public class DenseLayer extends Layer {

    public final ActivationFunction activationFunction;
    public final int outSize;
    public final int inSize;
    private final WeightInit weightInit;
    private final int parameterCount;

    public MatrixF weights;
    public VecF biases;
    public VecF zValueCache = new VecF();
    public VecF activationCache = new VecF();

    public DenseLayer(int inSize, int outSize, ActivationFunction activation, WeightInit weightInit) {
        super();

        this.outSize = outSize;
        this.inSize = inSize;
        this.weights = new MatrixF(inSize, outSize);
        this.biases = new VecF(outSize);
        this.activationFunction = activation;
        this.weightInit = weightInit;
        this.parameterCount = biases.getLength() + this.weights.getHeight() * this.weights.getWidth();
    }

    @Override
    public VecF processInput(VecF inputs) {

        if(inputs.getLength() != this.inSize)
            throw new IllegalStateException(String.format(
                    "DenseLayer received a larger amount of inputs (%d) than expected (%d)!", inputs.getLength(), this.inSize));


        VecF activation = this.weights.mult(inputs).add(this.biases);

        this.zValueCache = activation;
        this.activationCache = this.activationFunction.apply(activation);
        return this.activationCache;
    }

    @Override
    public void init(Random rand) {

        for(int i = 0; i < this.weights.getWidth(); i++)
            for(int j = 0; j < this.weights.getHeight(); j++)
                this.weights.set(i, j, this.weightInit.next(rand, this.inSize));//rand.nextFloat() * factor);

        for(int i = 0; i < this.biases.getLength(); i++)
            this.biases.set(i, this.weightInit.next(rand, this.inSize));//rand.nextFloat() * factor);
    }

    @Override
    public int getParameterCount() {
        return this.parameterCount;
    }

    @Override
    public float[] getParameters() {
        float[] params = new float[this.parameterCount];

        System.arraycopy(this.weights.toArray(), 0, params, 0, this.weights.getHeight() * this.weights.getWidth());
        System.arraycopy(this.biases.toArray(), 0, params, this.weights.getHeight() * this.weights.getWidth(), this.biases.getLength());

        return params;
    }

    @Override
    public void setParameters(float[] parameters) {
        this.weights.fromArray(Arrays.copyOfRange(parameters, 0, this.weights.getHeight() * this.weights.getWidth()));
        this.biases.fromArray(Arrays.copyOfRange(parameters, this.weights.getHeight() * this.weights.getWidth(), this.parameterCount));
    }
}
