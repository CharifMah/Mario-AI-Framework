import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import engine.core.MarioLevelModel;

public class LevelMatrixConverter {
    private static final String LevelPath = "./levels/generated/FlatGroundGenerator/10.txt";
    private static final int MAX_JUMP = 6;
    private static final int MAX_JUMP_X = 6; // portée horizontale du saut oblique

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

            int fallY = simulateFall(level, x, y, width, height);
            if (fallY == -1 || visited[x][fallY]) continue;

            // Marquer la case comme accessible
            visited[x][fallY] = true;

            // Marche horizontale illimitée à droite
            int nx = x + 1;
            while (inBounds(nx, fallY, width, height)
                    && isWalkable(level, nx, fallY)
                    && inBounds(nx, fallY + 1, width, height)
                    && isSolid(level, nx, fallY + 1)
                    && !visited[nx][fallY]) {
                visited[nx][fallY] = true;
                queue.add(new int[]{nx, fallY});
                nx++;
            }
            // Marche horizontale illimitée à gauche
            nx = x - 1;
            while (inBounds(nx, fallY, width, height)
                    && isWalkable(level, nx, fallY)
                    && inBounds(nx, fallY + 1, width, height)
                    && isSolid(level, nx, fallY + 1)
                    && !visited[nx][fallY]) {
                visited[nx][fallY] = true;
                queue.add(new int[]{nx, fallY});
                nx--;
            }

            // Saut oblique (depuis le sol)
            if (inBounds(x, fallY + 1, width, height) && isSolid(level, x, fallY + 1)) {
                for (int h = 1; h <= MAX_JUMP; h++) {
                    for (int dx = -MAX_JUMP_X; dx <= MAX_JUMP_X; dx++) {
                        int sx = x + dx;
                        int sy = fallY - h;
                        if (!inBounds(sx, sy, width, height)) continue;
                        if (!isWalkable(level, sx, sy)) continue;

                        int finalY = simulateFall(level, sx, sy, width, height);
                        if (finalY != -1 && !visited[sx][finalY]) {
                            queue.add(new int[]{sx, finalY});
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
        while (inBounds(x, currentY + 1, width, height)) {
            if (isSolid(level, x, currentY + 1)) {
                break;
            }
            if (!isWalkable(level, x, currentY + 1)) {
                break;
            }
            currentY++;
        }
        // Autoriser la dernière ligne comme case de sol si elle est solide
        if (!inBounds(x, currentY, width, height) || !isWalkable(level, x, currentY)) {
            return -1;
        }
        if (currentY == height - 1 && isSolid(level, x, currentY)) {
            return currentY;
        }
        if (!inBounds(x, currentY + 1, width, height) || !isSolid(level, x, currentY + 1)) {
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

    // CORRIGÉ : le sol '1' est walkable !
    public static boolean isWalkable(MarioLevelModel level, int x, int y) {
        char c = level.getBlock(x, y);
        if (c == '1') return true; // le sol est walkable
        // Ajoute ici d'autres caractères marchables si besoin
        for (char block : MarioLevelModel.getBlockTiles()) {
            if (c == block) return false;
        }
        return true;
    }

    // CORRIGÉ : le sol '1' est solide !
    public static boolean isSolid(MarioLevelModel level, int x, int y) {
        char c = level.getBlock(x, y);
        if (c == '1') return true; // le sol est solide
        for (char block : MarioLevelModel.getBlockTiles()) {
            if (c == block) return true;
        }
        if (c == MarioLevelModel.PLATFORM) return true;
        return false;
    }

    public static boolean inBounds(int x, int y, int width, int height) {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    /**
     * Vérifie si la fin du niveau est accessible :
     * - Si 'F' existe, vérifie si elle est accessible.
     * - Sinon, vérifie si la dernière case solide de la dernière colonne est accessible.
     */
    public static boolean isLevelEndReachable(MarioLevelModel level) {
        int width = level.getWidth();
        int height = level.getHeight();
        int exitX = -1, exitY = -1;
        boolean foundExit = false;

        // Cherche la position de la sortie ('F')
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (level.getBlock(x, y) == MarioLevelModel.MARIO_EXIT) {
                    exitX = x;
                    exitY = y;
                    foundExit = true;
                    break;
                }
            }
            if (foundExit) break;
        }

        // Calcul de la matrice d'accessibilité
        boolean[][] accessibility = computeAccessibility(level);

        if (foundExit) {
            // La sortie est-elle accessible ?
            return accessibility[exitX][exitY];
        } else {
            // Cherche la dernière case solide de la dernière colonne (du bas vers le haut)
            int lastCol = width - 1;
            for (int y = height - 1; y >= 0; y--) {
                if (isSolid(level, lastCol, y)) {
                    // On vérifie si elle est accessible
                    return accessibility[lastCol][y];
                }
            }
            // Aucun sol trouvé dans la dernière colonne
            return false;
        }
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

        if (pShowMatrix)
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
        printStructuralCoherence(LevelPath, true);

        String levelString = getLevel(LevelPath);

        String[] lines = levelString.split("\n");
        int width = lines[0].length();
        int height = lines.length;

        MarioLevelModel level = new MarioLevelModel(width, height);
        level.copyFromString(levelString);

        boolean endReachable = LevelMatrixConverter.isLevelEndReachable(level);
        System.out.println("Fin du niveau accessible : " + (endReachable ? "OUI" : "NON"));
    }
}
