package levelGenerators.GAN;

import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class LevelGenerator implements MarioLevelGenerator {

    private static final String MODEL_PATH = "models/GAN/gan_savedmodel";
    private static final String MAPPING_PATH = "models/GAN/gan_mapping.json";
    private static final int HEIGHT = 16;
    private static final int WIDTH = 150;
    private static final int LATENT_DIM = 32;

    private SavedModelBundle model;
    private Map<Integer, Character> intToChar;
    private int nSymbols;
    private Random random;

    @SuppressWarnings("unchecked")
    public LevelGenerator() {
        try {
            // Chargement du modèle TensorFlow SavedModel
            model = SavedModelBundle.load(MODEL_PATH, "serve");
            random = new Random();

            // Chargement du mapping int->char
            ObjectMapper mapper = new ObjectMapper();
            Map<String, Object> mapping = mapper.readValue(new File(MAPPING_PATH), Map.class);

            intToChar = new HashMap<>();
            Map<String, Object> intToCharRaw = (Map<String, Object>) mapping.get("int_to_char");
            for (Map.Entry<String, Object> entry : intToCharRaw.entrySet()) {
                intToChar.put(Integer.parseInt(entry.getKey()), ((String) entry.getValue()).charAt(0));
            }
            nSymbols = intToChar.size();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public String getGeneratedLevel(MarioLevelModel model, MarioTimer timer) {
        model.clearMap();
        float[] z = new float[LATENT_DIM];
        for (int i = 0; i < LATENT_DIM; i++) {
            z[i] = (float) random.nextGaussian();
        }
        // Format batch [1, LATENT_DIM]
        try (TFloat32 zTensor = TFloat32.tensorOf(Shape.of(1, LATENT_DIM))) {
            for (int i = 0; i < LATENT_DIM; i++) {
                zTensor.setFloat(z[i], 0, i);
            }

            // Inference TensorFlow GAN
            Tensor output = this.model.session()
                    .runner()
                    .feed("keras_tensor", zTensor)
                    .fetch("StatefulPartitionedCall") // Nom du node à adapter si besoin
                    .run().get(0);

            TFloat32 outT = (TFloat32) output;
            var nd = outT.asNdArray(); // NDArray 4D [1, HEIGHT, WIDTH, nSymbols]

            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    int maxIdx = 0;
                    float maxVal = nd.getFloat(0, y, x, 0);
                    for (int k = 1; k < nSymbols; k++) {
                        float val = nd.getFloat(0, y, x, k);
                        if (val > maxVal) {
                            maxVal = val;
                            maxIdx = k;
                        }
                    }
                    char tile = intToChar.getOrDefault(maxIdx, '-');
                    model.setBlock(x, y, tile);
                }
            }
            output.close();
        }
        return model.getMap();
    }

    @Override
    public String getGeneratorName() {
        return "GANGeneratorTF";
    }
}
