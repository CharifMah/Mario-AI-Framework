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
	public static final String BasePath = "levels/generated/";
	
    public static void printResults(MarioResult result, String pFilePath) {
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
        System.out.println("**************************************************************");
        LevelMatrixConverter.printStructuralCoherence(pFilePath, true);
        
    }

    public static void main(String[] args) {
    	if (LevelConfig.isBatchPlay) {
    		for (int i = 0; i < LevelConfig.Iteration; i++) {
				Generate();
			}
		}
    	else
			Generate();
    }
    
    public static void Generate()
    {
        MarioLevelGenerator generator = LevelConfig.Generator;
        String levelString = generator.getGeneratedLevel(new MarioLevelModel(150, 16), new MarioTimer(5 * 60 * 60 * 1000));

        String lLevelPath = WriteMap(levelString,generator);

        MarioGame game = new MarioGame();
        MarioResult lResult = game.runGame(LevelConfig.Agent, levelString, 20, 0, true);
        printResults(lResult,lLevelPath);
        
        MarioLevelModel level = new MarioLevelModel(150, 16);
        level.copyFromString(levelString);
        
        PlayLevel.logResultsToCSV(lResult, level, lLevelPath,"./logen.csv");
    }

    public static String WriteMap(String pLevel, MarioLevelGenerator pGenerator) {
        String baseName = pGenerator.getGeneratorName();
        String dirPath =  BasePath + baseName + '/';
        String lPath = "";
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

            lPath = dirPath + nextIndex + ".txt";
            FileWriter writer = new FileWriter(lPath);
            writer.write(pLevel);
            writer.close();
            System.out.println("Niveau sauvegardé dans " + lPath);

        } catch (IOException e) {
            System.err.println("Erreur lors de la sauvegarde du niveau : " + e.getMessage());
        }
        
        return lPath;
    }
}
