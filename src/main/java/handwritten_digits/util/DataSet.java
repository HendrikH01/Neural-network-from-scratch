package handwritten_digits.util;

import handwritten_digits.math.VecF;

import java.util.Arrays;
import java.util.Iterator;

public class DataSet implements Iterator<Example<VecF, VecF>> {

    public final int size;
    private int index = 0;
    private int batchIndex = 0;
    private final VecF[] features;
    private final VecF[] labels;

    public DataSet(VecF[] features, VecF[] labels) {
        if(features.length != labels.length)
            throw new IllegalArgumentException("Feature and label arrays must be the same size!");
        
        this.size = features.length;
        this.features = features;
        this.labels = labels;
    }

    public void reset() {
        this.index = 0;
        this.batchIndex = 0;
    }

    public boolean hasNextBatch() {
        return this.batchIndex + 1 < this.size;
    }

    public DataSet nextBatch(int size) {
        int s = Math.min(size, this.size - this.batchIndex) - 1;

        DataSet batch = new DataSet(Arrays.copyOfRange(this.features, batchIndex, batchIndex + s),
                Arrays.copyOfRange(this.labels, batchIndex, batchIndex + s));

        this.batchIndex += s;
        return batch;
    }

    @Override
    public boolean hasNext() {
        return this.index + 1 < this.size;
    }

    @Override
    public Example<VecF, VecF> next() {
        Example<VecF, VecF> pair = new Example<>(this.features[this.index], this.labels[this.index]);
        this.index++;

        return pair;
    }
}
