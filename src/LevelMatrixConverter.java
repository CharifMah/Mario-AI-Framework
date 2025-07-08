
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import engine.core.MarioLevelModel;

public class LevelMatrixConverter {

    private static final int MAX_JUMP = 6;
    private static final int MAX_JUMP_X = 4; // portée horizontale du saut oblique (3+1)

    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
            System.err.println("Erreur de lecture du niveau : " + filepath);
        }
        return content;
    }

    public static boolean[][] computeAccessibility(MarioLevelModel level) {
        int width = level.getWidth();
        int height = level.getHeight();
        boolean[][] visited = new boolean[width][height];

        int[] start = findMarioStart(level);
        if (start == null) return visited;

        Queue<int[]> queue = new LinkedList<>();
        queue.add(start);

        while (!queue.isEmpty()) {
            int[] pos = queue.poll();
            int x = pos[0], y = pos[1];

            // --- Propagation de la chute depuis chaque nouvelle case ---
            int fallY = simulateFall(level, x, y, width, height);
            if (fallY == -1 || visited[x][fallY]) continue;
            visited[x][fallY] = true;

            // Marche gauche/droite si sol sous les pieds
            if (inBounds(x, fallY + 1, width, height) && isSolid(level, x, fallY + 1)) {
                // Exploration horizontale plus extensive
                for (int dx : new int[]{-1, 1}) {
                    // Essayer de marcher plusieurs cases dans la même direction
                    for (int steps = 1; steps <= 10; steps++) {
                        int nx = x + (dx * steps);
                        int ny = fallY;
                        
                        if (!inBounds(nx, ny, width, height)) break;
                        if (!isWalkable(level, nx, ny)) break;
                        
                        // Simuler la chute après avoir marché
                        int finalY = simulateFall(level, nx, ny, width, height);
                        if (finalY != -1 && !visited[nx][finalY]) {
                            queue.add(new int[]{nx, finalY});
                        }
                    }
                }
            }

            // Saut oblique (depuis le sol)
            if (inBounds(x, fallY + 1, width, height) && isSolid(level, x, fallY + 1)) {
                for (int h = 1; h <= MAX_JUMP; h++) {
                    for (int dx = -MAX_JUMP_X; dx <= MAX_JUMP_X; dx++) {
                        int nx = x + dx;
                        int ny = fallY - h;
                        if (!inBounds(nx, ny, width, height)) continue;
                        if (!isWalkable(level, nx, ny)) continue;
                        
                        // Simuler la chute après le saut
                        int finalY = simulateFall(level, nx, ny, width, height);
                        if (finalY != -1 && !visited[nx][finalY]) {
                            queue.add(new int[]{nx, finalY});
                        }
                    }
                }
            }
        }
        return visited;
    }

    /**
     * Simule la chute de Mario depuis une position donnée
     * @param level le niveau
     * @param x position x de départ
     * @param y position y de départ
     * @param width largeur du niveau
     * @param height hauteur du niveau
     * @return la position Y finale après chute, ou -1 si hors limites
     */
    private static int simulateFall(MarioLevelModel level, int x, int y, int width, int height) {
        int currentY = y;
        
        // Tomber jusqu'à trouver un sol ou sortir des limites
        while (inBounds(x, currentY + 1, width, height)) {
            if (isSolid(level, x, currentY + 1)) {
                // On a trouvé un sol, on s'arrête sur la case au-dessus
                break;
            }
            if (!isWalkable(level, x, currentY + 1)) {
                // On ne peut pas traverser cette case (obstacle)
                break;
            }
            currentY++;
        }
        
        // Vérifier si la position finale est valide
        if (!inBounds(x, currentY, width, height) || !isWalkable(level, x, currentY)) {
            return -1;
        }
        
        // Vérifier qu'il y a bien un sol sous la position finale
        if (!inBounds(x, currentY + 1, width, height) || !isSolid(level, x, currentY + 1)) {
            // Pas de sol sous les pieds, Mario tomberait dans le vide
            return -1;
        }
        
        return currentY;
    }

    public static int[] findMarioStart(MarioLevelModel level) {
        int width = level.getWidth();
        int height = level.getHeight();
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                if (level.getBlock(x, y) == MarioLevelModel.MARIO_START)
                    return new int[]{x, y};
        // Sinon, cherche le premier sol en bas à gauche
        for (int x = 0; x < width; x++)
            for (int y = height - 1; y >= 0; y--)
                if (isSolid(level, x, y) && inBounds(x, y - 1, width, height))
                    return new int[]{x, y - 1};
        return null;
    }

    public static boolean isWalkable(MarioLevelModel level, int x, int y) {
        char c = level.getBlock(x, y);
        for (char block : MarioLevelModel.getBlockTiles()) {
            if (c == block) return false;
        }
        return true;
    }

    public static boolean isSolid(MarioLevelModel level, int x, int y) {
        char c = level.getBlock(x, y);
        for (char block : MarioLevelModel.getBlockTiles()) {
            if (c == block) return true;
        }
        if (c == MarioLevelModel.PLATFORM) return true;
        return false;
    }

    public static boolean inBounds(int x, int y, int width, int height) {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    public static void printAccessibility(boolean[][] matrix) {
        int width = matrix.length;
        int height = matrix[0].length;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                System.out.print(matrix[x][y] ? "1" : "0");
            }
            System.out.println();
        }
    }
    
    public static double printStructuralCoherence(String pFilepath, boolean pShowMatrix) {
        String levelString = getLevel(pFilepath);

        String[] lines = levelString.split("\n");
        int width = lines[0].length();
        int height = lines.length;

        MarioLevelModel level = new MarioLevelModel(width, height);
        level.copyFromString(levelString);

        boolean[][] accessibility = computeAccessibility(level);
        
        if(pShowMatrix)
        printAccessibility(accessibility);

        int accessible = 0, total = 0;
        for (int x = 0; x < level.getWidth(); x++)
            for (int y = 0; y < level.getHeight(); y++) {
                if (isSolid(level, x, y)) total++;
                if (accessibility[x][y]) accessible++;
            }
        double coherence = total > 0 ? (double) accessible / total : 0.0;
        System.out.println("Cohérence structurelle : " + coherence);
        return coherence;
    }


    public static void main(String[] args) {
        String filepath = "./levels/generated/FlatGroundGenerator/10.txt";
        printStructuralCoherence(filepath, true);
    }
}