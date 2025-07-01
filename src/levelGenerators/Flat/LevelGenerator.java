package levelGenerators.Flat;

import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;
/**
 * Author : Mahmoud Charif
 * Genere un niveau plat
 * 2025-07-01
 */

public class LevelGenerator implements MarioLevelGenerator {

    @Override
    public String getGeneratedLevel(MarioLevelModel model, MarioTimer timer) {
        model.clearMap();
        int groundHeight = model.getHeight() - 1; // Dernière ligne = sol

        // On place des blocs de sol ('X') sur toute la largeur
        for (int x = 0; x < model.getWidth(); x++) {
            model.setBlock(x, groundHeight, 'X');
        }

        return model.getMap();
    }

    @Override
    public String getGeneratorName() {
        return "FlatGroundGenerator";
    }
}
