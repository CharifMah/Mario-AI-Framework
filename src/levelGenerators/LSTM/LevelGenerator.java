package levelGenerators.LSTM;

import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.types.TFloat32;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class LevelGenerator implements MarioLevelGenerator {

    private static final String MODEL_PATH = "models/LSTMs/mario_lstm_savedmodel";
    private static final String MAPPING_PATH = "models/LSTMs/char_mapping.json";
    private static final int SEQUENCE_LENGTH = 100;

    private SavedModelBundle model;
    private Map<Character, Integer> charToInt;
    private Map<Integer, Character> intToChar;
    private int nVocab;
    private String inputTensorName = "serving_default_input_1:0";
    private String outputTensorName = "StatefulPartitionedCall:0";

    @SuppressWarnings("unchecked")
    public LevelGenerator() {
        try {
            // Chargement du modèle TensorFlow SavedModel
            model = SavedModelBundle.load(MODEL_PATH, "serve");

            // Chargement du mapping caractères <-> entiers
            ObjectMapper mapper = new ObjectMapper();
            Map<String, Object> mapping = mapper.readValue(new File(MAPPING_PATH), Map.class);

            charToInt = new HashMap<>();
            intToChar = new HashMap<>();

            Map<String, Object> charToIntRaw = (Map<String, Object>) mapping.get("char_to_int");
            for (Map.Entry<String, Object> entry : charToIntRaw.entrySet()) {
                charToInt.put(entry.getKey().charAt(0), ((Number) entry.getValue()).intValue());
            }
            Map<String, Object> intToCharRaw = (Map<String, Object>) mapping.get("int_to_char");
            for (Map.Entry<String, Object> entry : intToCharRaw.entrySet()) {
                intToChar.put(Integer.parseInt(entry.getKey()), ((String) entry.getValue()).charAt(0));
            }
            nVocab = charToInt.size();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public String getGeneratedLevel(MarioLevelModel model, MarioTimer timer) {
        model.clearMap();
        int width = model.getWidth();
        int height = model.getHeight();

        // Génère un seed initial (par ex, le premier caractère du mapping)
        StringBuilder seed = new StringBuilder();
        for (int i = 0; i < SEQUENCE_LENGTH; i++) {
            seed.append(intToChar.get(0));
        }

        // Génère la séquence avec le modèle LSTM
        String generated = generateLevel(seed.toString(), width * height);

        // Remplit la carte Mario à partir de la chaîne générée
        int idx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (idx < generated.length()) {
                    model.setBlock(x, y, generated.charAt(idx));
                    idx++;
                }
            }
        }
        return model.getMap();
    }

    public String generateLevel(String seed, int levelLength) {
        StringBuilder generated = new StringBuilder(seed);
        for (int i = 0; i < levelLength; i++) {
            float[][][] inputArr = new float[1][SEQUENCE_LENGTH][1];
            for (int j = 0; j < SEQUENCE_LENGTH; j++) {
                int idx = Math.max(0, generated.length() - SEQUENCE_LENGTH + j);
                char c = generated.charAt(idx);
                int charIdx = charToInt.getOrDefault(c, 0);
                inputArr[0][j][0] = (float) charIdx / nVocab;
            }
            try (TFloat32 inputTensor = TFloat32.tensorOf(Shape.of(1, SEQUENCE_LENGTH, 1))) {
                for (int j = 0; j < SEQUENCE_LENGTH; j++) {
                    inputTensor.setFloat(inputArr[0][j][0], 0, j, 0);
                }
                // Inference TensorFlow
                Tensor output = model.session()
                        .runner()
                        .feed(inputTensorName, inputTensor)
                        .fetch(outputTensorName)
                        .run().get(0);

                TFloat32 outputT = (TFloat32) output;
                FloatNdArray nd = NdArrays.ofFloats(Shape.of(1, nVocab));
                outputT.copyTo(nd);

                // Trouver l'indice du max dans la sortie softmax [1, nVocab]
                float maxVal = -Float.MAX_VALUE;
                int maxIdx = 0;
                for (int k = 0; k < nVocab; k++) {
                    float val = nd.getFloat(0, k);
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = k;
                    }
                }
                generated.append(intToChar.get(maxIdx));
                output.close();
            }
        }
        return generated.substring(SEQUENCE_LENGTH); // On ignore le seed initial
    }

    @Override
    public String getGeneratorName() {
        return "LSTMGeneratorTF";
    }
}
