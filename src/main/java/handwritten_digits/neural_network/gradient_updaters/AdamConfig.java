package handwritten_digits.neural_network.gradient_updaters;

public class AdamConfig implements IGradientUpdaterConfig {

    public static final AdamConfig DEFAULT_INSTANCE =
            new AdamConfig(0.001F, 0.9F, 0.99F, 0.00000001F);

    public final float alpha;
    public final float beta1;
    public final float beta2;
    public final float epsilon;

    public AdamConfig(float alpha, float beta1, float beta2, float epsilon) {
        this.alpha = alpha;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }
}
