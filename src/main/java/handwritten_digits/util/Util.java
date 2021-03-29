package handwritten_digits.util;

import handwritten_digits.math.VecF;

public class Util {

    public static int getLargestElementIndex(VecF vec) {

        int largest = 0;

        for(int i = 0; i < vec.getLength(); i++) {
            largest = vec.get(largest) < vec.get(i) ? i : largest;
        }

        return largest;
    }

    public static float getLargestElement(VecF vec) {

        float largest = Float.MIN_VALUE;

        for(int i = 0; i < vec.getLength(); i++) {
            largest = Math.max(largest, vec.get(i));
        }

        return largest;
    }

    public static float clamp(float f, float min, float max) {
        return f > max ? max : Math.max(f, min);
    }
}
