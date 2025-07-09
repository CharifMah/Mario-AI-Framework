import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import engine.core.MarioGame;
import engine.core.MarioLevelModel;
import engine.core.MarioResult;

public class PlayLevel {

    public static void main(String[] args) {
        MarioGame game = new MarioGame();
        // printResults(game.playGame(getLevel("../levels/original/lvl-1.txt"), 200, 0));
        MarioResult lResult = game.runGame(LevelConfig.Agent, getLevel(LevelConfig.LevelPath), 20, 0, true);
        printResults(lResult);
        
        String levelString = getLevel(LevelConfig.LevelPath);
        if (levelString == null || levelString.isEmpty()) {
            System.err.println("Niveau vide ou introuvable.");
            return;
        }
        
        MarioLevelModel level = new MarioLevelModel(150, 16);
        level.copyFromString(levelString);
        
        logResultsToCSV(lResult,level ,"./log.csv");
    }
    
	public static void printResults(MarioResult result) {
	    System.out.println("============================================================");
	    System.out.println("                      MARIO GAME RESULTS / " + LevelConfig.Agent.getAgentName() + " / " + LevelConfig.LevelPath );
	    System.out.println("============================================================");
	    System.out.printf("%-25s : %s\n", "Game Status", result.getGameStatus());
	    System.out.printf("%-25s : %.1f %%\n", "Completion", 100 * result.getCompletionPercentage());
	    System.out.printf("%-25s : %d\n", "Lives", result.getCurrentLives());
	    System.out.printf("%-25s : %d\n", "Coins", result.getCurrentCoins());
	    System.out.printf("%-25s : %ds\n", "Time Left", (int)Math.ceil(result.getRemainingTime() / 1000f));
	    System.out.println("------------------------------------------------------------");
	    System.out.printf("%-25s : %d\n", "Mushrooms Collected", result.getNumCollectedMushrooms());
	    System.out.printf("%-25s : %d\n", "Fire Flowers Collected", result.getNumCollectedFireflower());
	    System.out.printf("%-25s : %d\n", "Coins Collected (tiles)", result.getNumCollectedTileCoins());
	    System.out.println("------------------------------------------------------------");
	    System.out.printf("%-25s : %d\n", "Bricks Destroyed", result.getNumDestroyedBricks());
	    System.out.printf("%-25s : %d\n", "Jumps", result.getNumJumps());
	    System.out.printf("%-25s : %.2f\n", "Max X Jump", result.getMaxXJump());
	    System.out.printf("%-25s : %d\n", "Max Jump Air Time", result.getMaxJumpAirTime());
	    System.out.println("------------------------------------------------------------");
	    System.out.printf("%-25s : %d\n", "Kills (Total)", result.getKillsTotal());
	    System.out.printf("%-25s : %d\n", "Kills by Stomp", result.getKillsByStomp());
	    System.out.printf("%-25s : %d\n", "Kills by Fire", result.getKillsByFire());
	    System.out.printf("%-25s : %d\n", "Kills by Shell", result.getKillsByShell());
	    System.out.printf("%-25s : %d\n", "Kills by Fall", result.getKillsByFall());
	    System.out.println("============================================================");
	    LevelMatrixConverter.printStructuralCoherence(LevelConfig.LevelPath, true);
	    MarioLevelIndicators.printMapIndicator(LevelConfig.LevelPath);
	}
	
	public static void logResultsToCSV(MarioResult result, MarioLevelModel level, String filePath) {
	    boolean fileExists = new File(filePath).exists();

	    // Calcul des métriques
	    double coherence = LevelMatrixConverter.printStructuralCoherence(LevelConfig.LevelPath, false); // getter, pas print
	    double leniency = MarioLevelIndicators.computeLeniency(level);
	    double linearity = MarioLevelIndicators.computeLinearity(level);
	    int groundSegments = MarioLevelIndicators.computeGroundSegments(level);
	    double structuralDiversity = MarioLevelIndicators.computeStructuralDiversity(level);
	    int unjumpableHoles = MarioLevelIndicators.countUnjumpableHoles(level, 6, 6);
	    int totalVerticalHoles = MarioLevelIndicators.countTotalVerticalHoles(level);

	    // Détail des trous verticaux >= 6 de large
	    List<int[]> holes = MarioLevelIndicators.findVerticalHoles(level);
	    List<String> largeHolesDetail = new ArrayList<>();
	    for (int[] hole : holes) {
	        int width = hole[1] - hole[0];
	        if (width >= 6) {
	            largeHolesDetail.add(String.format("[%d-%d:%d]", hole[0], hole[1] - 1, width));
	        }
	    }
	    String largeHolesString = String.join(";", largeHolesDetail);

	    try (PrintWriter writer = new PrintWriter(new FileWriter(filePath, true))) {
	        // Écrire l'en-tête une seule fois
	        if (!fileExists) {
	            writer.println(
	                "LevelPath,AgentName," +
	                "GameStatus,Completion,Lives,Coins,TimeLeft," +
	                "Mushrooms,FireFlowers,TileCoins,BricksDestroyed,Jumps,MaxXJump,MaxJumpAirTime," +
	                "KillsTotal,KillsStomp,KillsFire,KillsShell,KillsFall," +
	                "Coherence,Leniency,Linearity,GroundSegments,StructuralDiversity," +
	                "UnjumpableHoles,TotalVerticalHoles,LargeVerticalHolesDetail"
	            );
	        }

	        // Ligne de données, toujours dans le même ordre que l'en-tête
	        writer.printf(Locale.US,
	        	    "\"%s\"," +      // LevelPath
    	    		"\"%s\"," +      // AgentName
	        	    "%s,%.2f,%d,%d,%d," +      
	        	    "%d,%d,%d,%d,%d,%.2f,%d," +
	        	    "%d,%d,%d,%d,%d," +
	        	    "%.2f,%.0f,%.6f,%d,%.2f," +
	        	    "%d,%d,\"%s\"\n",
	        	    LevelConfig.LevelPath,
	        	    LevelConfig.Agent.getAgentName(),
	        	    result.getGameStatus(),
	        	    100 * result.getCompletionPercentage(),
	        	    result.getCurrentLives(),
	        	    result.getCurrentCoins(),
	        	    (int)Math.ceil(result.getRemainingTime() / 1000f),
	        	    result.getNumCollectedMushrooms(),
	        	    result.getNumCollectedFireflower(),
	        	    result.getNumCollectedTileCoins(),
	        	    result.getNumDestroyedBricks(),
	        	    result.getNumJumps(),
	        	    result.getMaxXJump(),
	        	    result.getMaxJumpAirTime(),
	        	    result.getKillsTotal(),
	        	    result.getKillsByStomp(),
	        	    result.getKillsByFire(),
	        	    result.getKillsByShell(),
	        	    result.getKillsByFall(),
	        	    coherence,
	        	    leniency,
	        	    linearity,
	        	    groundSegments,
	        	    structuralDiversity,
	        	    unjumpableHoles,
	        	    totalVerticalHoles,
	        	    largeHolesString
	        	);
	    } catch (IOException e) {
	        e.printStackTrace();
	    }
	}
	
    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
        }
        return content;
    }

}
