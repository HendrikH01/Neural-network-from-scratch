package handwritten_digits.analysis;

import handwritten_digits.math.VecF;
import handwritten_digits.util.Util;
import org.apache.commons.lang3.StringUtils;

public class ClassificationEvaluator implements IEvaluator {

    private final int classes;
    private int[][] classificationResults;
    private int examples = 0;
    private int correct = 0;

    public ClassificationEvaluator(int classes) {
        this.classes = classes;
        classificationResults = new int[classes][classes];
    }

    @Override
    public boolean compareResult(VecF output, VecF expected) {
        if(output.getLength() != this.classes ||expected.getLength() != this.classes) {
            throw new IllegalStateException("ClassificationEvaluator received Vectors of wrong length");
        }
        examples++;

        int out = Util.getLargestElementIndex(output);
        int exp = Util.getLargestElementIndex(expected);

        this.classificationResults[out][exp]++;

        if(out == exp) {
            correct++;
            return true;
        }

        return false;
    }

    @Override
    public void printStats() {
        System.out.printf("Guessed right: %.2f%% of the time!\n\n", 100.0 * correct / (double) examples);

        System.out.println("Classification results: \n");

        for (int j = 0; j < this.classes; j++)
            System.out.print("########");

        System.out.println("#");
        for(int i = 0; i < this.classes; i++) {
            for (int j = 0; j < this.classes; j++) {
                System.out.print("# " + StringUtils.rightPad(this.classificationResults[i][j] + " ", 6, ' '));

            }

            System.out.println("#");

            for (int j = 0; j < this.classes; j++)
                System.out.print("########");

            System.out.println("#");
        }
    }
}
