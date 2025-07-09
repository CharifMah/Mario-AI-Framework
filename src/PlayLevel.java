import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import engine.core.MarioGame;
import engine.core.MarioResult;

public class PlayLevel {
	public static final String LevelPath = "./levels/generated/GANGeneratorTF/9.txt";

	public static void printResults(MarioResult result) {
	    System.out.println("============================================================");
	    System.out.println("                      MARIO GAME RESULTS                    ");
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
	    LevelMatrixConverter.printStructuralCoherence(LevelPath, true);
	}


    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
        }
        return content;
    }

    public static void main(String[] args) {
        MarioGame game = new MarioGame();
        // printResults(game.playGame(getLevel("../levels/original/lvl-1.txt"), 200, 0));
        printResults(game.runGame(new agents.andySloane.Agent(), getLevel(LevelPath), 20, 0, true));
    }
}
