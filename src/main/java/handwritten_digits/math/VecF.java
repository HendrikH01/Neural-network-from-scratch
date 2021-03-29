package handwritten_digits.math;

import handwritten_digits.util.Util;

import java.util.Arrays;

public class VecF implements ITensorF {

    float[] values;

    public VecF(float... values) {
        this.values = values;
    }

    public VecF(int length) {
        this.values = new float[length];
    }

    public VecF(VecF vec) {
        this.values = vec.values.clone();
    }

    public VecF(int length, float f) {
        this.values = new float[length];

        for(int i = 0; i < length; i++)
            this.values[i] = f;
    }

    public float get(int i) {
        return this.values[i];
    }

    public void set(int i, float f) {
        this.values[i] = f;
    }

    public VecF add(VecF vec) {
        VecF out = new VecF(this);

        for(int i = 0; i < this.values.length; i++) {
            out.values[i] += vec.values[i];
        }

        return out;
    }

    public VecF sub(VecF vec) {
        VecF out = new VecF(this);

        for(int i = 0; i < this.values.length; i++) {
            out.values[i] -= vec.values[i];
        }

        return out;
    }

    public VecF mult(VecF vec) {
        VecF out = new VecF(this);

        for(int i = 0; i < this.values.length; i++) {
            out.values[i] *= vec.values[i];
        }

        return out;
    }

    public VecF div(VecF vec) {
        VecF out = new VecF(this);

        for(int i = 0; i < this.values.length; i++) {
            if(vec.values[i] != 0)
                out.values[i] /= vec.values[i];
        }

        return out;
    }

    public VecF clamp(float min, float max) {
        VecF out = new VecF(this.getLength());

        for(int i = 0; i < this.values.length; i++) {
            out.values[i] = Util.clamp(this.get(i), min, max);
        }

        return out;
    }

    public VecF mult(double scalar) {
        VecF out = new VecF(this);
        for(int i = 0; i < this.values.length; i++) {
            out.values[i] *= scalar;
        }

        return out;
    }

    public VecF div(double scalar) {
        VecF out = new VecF(this);
        for(int i = 0; i < this.values.length; i++) {
            out.values[i] /= scalar;
        }

        return out;
    }

    public int getLength() {
        return this.values.length;
    }

    @Override
    public String toString() {
        return "VecF{" +
                "values=" + Arrays.toString(values) +
                '}';
    }

    @Override
    public float[] toArray() {
        return this.values;
    }

    @Override
    public void fromArray(float[] arr) {
        this.values = arr;
    }
}
