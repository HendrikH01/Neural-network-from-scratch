package handwritten_digits.neural_network.layers;

import handwritten_digits.math.VecF;

import java.util.Random;

public abstract class Layer {

    public Layer() {}

    public abstract VecF processInput(VecF inputs);

    public abstract void init(Random rand);

    public  int getParameterCount() {
        return 0;
    }

    public float[] getParameters() {
        return new float[0];
    }

    public void setParameters(float[] floats) {}
}
