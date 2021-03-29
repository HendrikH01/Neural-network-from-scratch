package handwritten_digits.math;

public class MatrixF implements ITensorF {

    private final int width;
    private final int height;
    private float[][] values;

    public MatrixF(int width, int height) {
        this.width = width;
        this.height = height;
        this.values = new float[width][height];
    }

    public MatrixF(int width, int height, float... values) {
        this.width = width;
        this.height = height;
        this.values = new float[width][height];

        for (int i = 0; i < values.length; i++) {
            this.values[i / width][i % width] = values[i];
        }
    }

    public MatrixF(MatrixF matrix) {
        this.width = matrix.width;
        this.height = matrix.height;
        this.values = matrix.values.clone();
    }

    public float get(int i, int j) {
        return this.values[i][j];
    }

    public void set(int i, int j, float f) {
        this.values[i][j] = f;
    }

    public int getWidth() {
        return this.width;
    }

    public int getHeight() {
        return this.height;
    }

    public MatrixF mult(MatrixF matrix) {
        if (this.height != matrix.width)
            throw new IllegalArgumentException(String.format("Matrix of shape (%d, %d) can't be multiplied with matrix of shape (%d, %d)",
                    this.width, this.height, matrix.width, matrix.height));

        MatrixF out = new MatrixF(this.width, matrix.height);

        for (int i = 0; i < out.width; i++) {
            for (int j = 0; j < out.height; j++) {
                float cell = 0;

                for (int k = 0; k < matrix.height; k++) {
                    cell += this.values[j][k] * matrix.values[k][i];
                }

                out.values[i][j] = cell;
            }
        }

        return out;
    }

    public VecF mult(VecF vec) {
        if (this.width != vec.getLength())
            throw new IllegalArgumentException(String.format("Matrix of width %d can't be multiplied with vector of length %d",
                    this.width, vec.getLength()));

        VecF out = new VecF(this.height);

        for(int j = 0; j < this.width; j++) {
            if(vec.get(j) == 0)
                continue;

            for(int i = 0; i < this.height; i++) {
                out.values[i] += this.values[j][i] * vec.get(j);
            }
        }

        return out;
    }

    public MatrixF mult(float scalar) {
        MatrixF out = new MatrixF(this.width, this.height);

        for(int i = 0; i < this.width; i++) {
            for(int j = 0; j < this.height; j++) {
                out.values[i][j] = this.values[i][j] * scalar;
            }
        }

        return out;
    }

    public MatrixF transpose() {
        MatrixF out = new MatrixF(this.height, this.width);

        for (int i = 0; i < out.width; i++) {
            for (int j = 0; j < out.height; j++) {
               out.values[i][j] = this.values[j][i];
            }
        }

        return out;
    }

    public MatrixF   sub(MatrixF matrix) {
        MatrixF out = new MatrixF(this.width, this.height);

        for(int i = 0; i < this.width; i++) {
            for(int j = 0; j < this.height; j++) {
                out.values[i][j] = this.values[i][j] - matrix.values[i][j];
            }
        }

        return out;
    }

    public MatrixF add(MatrixF matrix) {
        MatrixF out = new MatrixF(this.width, this.height);

        for(int i = 0; i < this.width; i++) {
            for(int j = 0; j < this.height; j++) {
                out.values[i][j] = this.values[i][j] + matrix.values[i][j];
            }
        }

        return out;
    }

    @Override
    public float[] toArray() {
        float[] arr = new float[width * this.height];

        for(int i = 0; i < this.width; i++) {
            for(int j = 0; j < this.height; j++) {
                arr[i + j * this.width] = this.values[i][j];
            }
        }

        return arr;
    }

    @Override
    public void fromArray(float[] arr) {
        for(int i = 0; i < this.width; i++) {
            for(int j = 0; j < this.height; j++) {
                this.values[i][j] = arr[i + j * this.width];
            }
        }
    }
}