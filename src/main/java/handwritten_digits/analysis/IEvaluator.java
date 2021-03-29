package handwritten_digits.analysis;

import handwritten_digits.math.VecF;

public interface IEvaluator {
    /**
     * return true if result was correct
     *
     * @param output
     * @param expected
     * @return
     */
    boolean compareResult(VecF output, VecF expected);

    void printStats();
}
