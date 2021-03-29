package handwritten_digits.util;

public class Example<F, S> {

    public final F feature;
    public final S label;

    public Example(F feature, S label) {
        this.feature = feature;
        this.label = label;
    }
}
