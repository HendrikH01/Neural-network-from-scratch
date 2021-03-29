package handwritten_digits.util;

import handwritten_digits.math.VecF;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.*;

public class MNISTLoader {

    public static DataSet loadMNIST(boolean train, long randomSeed) {
        List<VecF> images = new ArrayList<>();
        List<VecF> labels = new ArrayList<>();

        for(int i = 0; i < 10; i++) {
            try {
                List<VecF> vecs = convertToVector(getImages(i, train));
                images.addAll(vecs);

                VecF label = new VecF(10);
                label.set(i, 1.0F);
                List<VecF> labelsN = Arrays.asList(new VecF[vecs.size()]);
                Collections.fill(labelsN, label);
                labels.addAll(labelsN);

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        //shuffle
        Collections.shuffle(images, new Random(randomSeed));
        Collections.shuffle(labels, new Random(randomSeed));

        return new DataSet(images.toArray(new VecF[0]), labels.toArray(new VecF[0]));
    }

    public static List<VecF> convertToVector(List<byte[]> images) {
        List<VecF> vecs = new ArrayList<>();

        for(byte[] img : images) {
            VecF vec = new VecF(784);

            for(int i = 0; i < img.length; i++) {
                vec.set(i, ((float) (img[i] & 255)) / 255.0F);
            }

            vecs.add(vec);
        }

        return vecs;
    }

    public static List<byte[]> getImages(int number, boolean train) throws IOException {
        String path = "src/main/resources/mnist_png/" + (train ? "training/" : "testing/") + number;
        File folder = new File(path);

        String[] images = folder.list();
        List<byte[]> out = new ArrayList<>();
        assert images != null;

        for(String s : images) {
            BufferedImage bi = ImageIO.read(new File(path + "/" + s));
            out.add(((DataBufferByte)bi.getRaster().getDataBuffer()).getData());
        }

        return out;
    }

}
