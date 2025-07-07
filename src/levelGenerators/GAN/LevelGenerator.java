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
import java.util.List;

public class LevelGenerator implements MarioLevelGenerator {

    private static final String MODEL_PATH = "models/gan_lsi/gan_savedmodel";
    private static final String MAPPING_PATH = "models/gan_lsi/char_mapping.json";
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
            model = SavedModelBundle.load(MODEL_PATH, "serve");
            random = new Random();

            // ----------- Correction mapping int->char (ArrayList support) -----------
            ObjectMapper mapper = new ObjectMapper();
            Map<String, Object> mapping = mapper.readValue(new File(MAPPING_PATH), Map.class);

            intToChar = new HashMap<>();
            Object intToCharRawObj = mapping.get("int_to_char");
            if (intToCharRawObj instanceof List) {
                List<Object> intToCharList = (List<Object>) intToCharRawObj;
                for (int i = 0; i < intToCharList.size(); i++) {
                    intToChar.put(i, ((String) intToCharList.get(i)).charAt(0));
                }
            } else if (intToCharRawObj instanceof Map) {
                Map<String, Object> intToCharRaw = (Map<String, Object>) intToCharRawObj;
                for (Map.Entry<String, Object> entry : intToCharRaw.entrySet()) {
                    intToChar.put(Integer.parseInt(entry.getKey()), ((String) entry.getValue()).charAt(0));
                }
            } else {
                throw new IllegalArgumentException("Unknown int_to_char format in mapping JSON");
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

            // --------- Correction : noms exacts pour TensorFlow Java ---------
            Tensor output = this.model.session()
                    .runner()
                    .feed("serving_default_keras_tensor:0", zTensor)
                    .fetch("StatefulPartitionedCall_1:0")
                    .run().get(0);

            // Récupération des valeurs via getFloat
            TFloat32 outT = (TFloat32) output;
            for (int y = 0; y < HEIGHT; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    int maxIdx = 0;
                    float maxVal = outT.getFloat(0, y, x, 0);
                    for (int k = 1; k < nSymbols; k++) {
                        float val = outT.getFloat(0, y, x, k);
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
