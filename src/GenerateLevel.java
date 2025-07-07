import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import engine.core.MarioGame;
import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioResult;
import engine.core.MarioTimer;

public class GenerateLevel {
    public static void printResults(MarioResult result) {
        System.out.println("****************************************************************");
        System.out.println("Game Status: " + result.getGameStatus().toString() +
                " Percentage Completion: " + result.getCompletionPercentage());
        System.out.println("Lives: " + result.getCurrentLives() + " Coins: " + result.getCurrentCoins() +
                " Remaining Time: " + (int) Math.ceil(result.getRemainingTime() / 1000f));
        System.out.println("Mario State: " + result.getMarioMode() +
                " (Mushrooms: " + result.getNumCollectedMushrooms() + " Fire Flowers: " + result.getNumCollectedFireflower() + ")");
        System.out.println("Total Kills: " + result.getKillsTotal() + " (Stomps: " + result.getKillsByStomp() +
                " Fireballs: " + result.getKillsByFire() + " Shells: " + result.getKillsByShell() +
                " Falls: " + result.getKillsByFall() + ")");
        System.out.println("Bricks: " + result.getNumDestroyedBricks() + " Jumps: " + result.getNumJumps() +
                " Max X Jump: " + result.getMaxXJump() + " Max Air Time: " + result.getMaxJumpAirTime());
        System.out.println("****************************************************************");
    }

    public static void main(String[] args) {
        MarioLevelGenerator generator = new levelGenerators..LevelGenerator();
        String level = generator.getGeneratedLevel(new MarioLevelModel(150, 16), new MarioTimer(5 * 60 * 60 * 1000));

        WriteMap(level,generator);

        MarioGame game = new MarioGame();
        printResults(game.runGame(new agents.robinBaumgarten.Agent(), level, 20, 0, true));
    }
    

    public static void WriteMap(String pLevel, MarioLevelGenerator generator) {
        String baseName = generator.getGeneratorName();
        String dirPath = "levels/generated/" + baseName + '/';

        int nextIndex = 1;

        try {
            Files.createDirectories(Paths.get(dirPath));

            // Cherche le plus grand index existant dans ce dossier
            File dir = new File(dirPath);
            File[] files = dir.listFiles((d, name) -> name.endsWith(".txt"));
            if (files != null) {
                for (File f : files) {
                    String fname = f.getName().replace(".txt", "");
                    try {
                        int idx = Integer.parseInt(fname);
                        if (idx >= nextIndex) nextIndex = idx + 1;
                    } catch (NumberFormatException ignore) {}
                }
            }

            String filename = dirPath + nextIndex + ".txt";
            FileWriter writer = new FileWriter(filename);
            writer.write(pLevel);
            writer.close();
            System.out.println("Niveau sauvegardé dans " + filename);
        } catch (IOException e) {
            System.err.println("Erreur lors de la sauvegarde du niveau : " + e.getMessage());
        }
    }
}
