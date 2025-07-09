import engine.core.MarioLevelModel;
import engine.sprites.Mario;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;
import java.util.*;

public class MarioLevelIndicators {

    // --- Métriques ---

    // Leniency : score basé sur ennemis et obstacles
    public static int computeLeniency(MarioLevelModel level) {
        int score = 0;
        int w = level.getWidth();
        int h = level.getHeight();
        Set<Character> enemies = new HashSet<>();
        for (char c : MarioLevelModel.getEnemyCharacters()) enemies.add(c);
        Set<Character> obstacles = new HashSet<>();
        for (char c : MarioLevelModel.getBlockTiles()) obstacles.add(c);

        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                char c = level.getBlock(x, y);
                if (enemies.contains(c)) score -= 2;
                else if (obstacles.contains(c)) score -= 1;
            }
        }
        return score;
    }

    // Linearity : variance de la hauteur du sol, sur les colonnes où il y a du sol
    public static double computeLinearity(MarioLevelModel level) {
        int w = level.getWidth();
        int h = level.getHeight();
        
        List<Integer> groundHeights = new ArrayList<>();
        for (int x = 0; x < w; x++) {
            int y = findGroundHeight(level, x, h);
            if (y != -1) groundHeights.add(y);
        }
        
        if (groundHeights.isEmpty()) return 0.0;
        
        double mean = groundHeights.stream().mapToInt(i -> i).average().orElse(0);
        double var = 0;
        for (int y : groundHeights) var += Math.pow(y - mean, 2);
        return var / groundHeights.size();
    }

    // Trouve la hauteur du sol pour une colonne (ou -1 si pas de sol)
    private static int findGroundHeight(MarioLevelModel level, int x, int h) {
        for (int y = h-1; y >= 0; y--) {
            char c = level.getBlock(x, y);
            if (isGround(c)) 
            	return y;
        }
        return -1;
    }


    // Nombre de segments de sol sur la ligne la plus basse
    public static int computeGroundSegments(MarioLevelModel level) {
        int seg = 0;
        boolean inSeg = false;
        int y = level.getHeight() - 1;
        for (int x = 0; x < level.getWidth(); x++) {
            char c = level.getBlock(x, y);
            if (isGround(c)) {
                if (!inSeg) { seg++; inSeg = true; }
            } else {
                inSeg = false;
            }
        }
        return seg;
    }

    // Diversité structurelle : nombre de motifs 2x2 différents
    public static int computeStructuralDiversity(MarioLevelModel level) {
        Set<String> motifs = new HashSet<>();
        int w = level.getWidth();
        int h = level.getHeight();
        for (int x = 0; x < w - 1; x++) {
            for (int y = 0; y < h - 1; y++) {
                String motif = "" + level.getBlock(x, y) + level.getBlock(x+1, y)
                                 + level.getBlock(x, y+1) + level.getBlock(x+1, y+1);
                motifs.add(motif);
            }
        }
        return motifs.size();
    }

    // Utilitaire pour reconnaître le sol
    private static boolean isGround(char c) {
        return c == MarioLevelModel.GROUND
            || c == MarioLevelModel.PYRAMID_BLOCK
            || c == MarioLevelModel.PLATFORM
            || c == MarioLevelModel.NORMAL_BRICK
            || c == MarioLevelModel.COIN_BRICK
            || c == MarioLevelModel.LIFE_BRICK
            || c == MarioLevelModel.SPECIAL_BRICK
            || c == MarioLevelModel.SPECIAL_QUESTION_BLOCK
            || c == MarioLevelModel.COIN_QUESTION_BLOCK
            || c == MarioLevelModel.USED_BLOCK
            || c == MarioLevelModel.COIN_HIDDEN_BLOCK
            || c == MarioLevelModel.LIFE_HIDDEN_BLOCK
            || c == MarioLevelModel.PIPE
            || c == MarioLevelModel.PIPE_FLOWER
            || c == MarioLevelModel.BULLET_BILL;
    }

    // --- Méthode pour lire un niveau depuis un fichier ---
    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
            System.err.println("Erreur de lecture du niveau : " + filepath);
        }
        return content;
    }

    public static void printMapIndicator(String pLevelPath)
    {
        int width = 150;   // largeur du niveau
        int height = 16;   // hauteur du niveau
        int maxJumpDistance = 6; // distance max franchissable en un saut

        String levelString = getLevel(pLevelPath);
        if (levelString == null || levelString.isEmpty()) {
            System.err.println("Niveau vide ou introuvable.");
            return;
        }

        MarioLevelModel level = new MarioLevelModel(width, height);
        level.copyFromString(levelString);

        System.out.println("Leniency : " + computeLeniency(level));
        System.out.println("Linearity : " + computeLinearity(level));
        System.out.println("Ground Segments : " + computeGroundSegments(level));
        System.out.println("Structural Diversity : " + computeStructuralDiversity(level));
    }

    // --- Exemple d'utilisation ---
    public static void main(String[] args) {
        // Paramètres à adapter selon tes niveaux
        String path = "./levels/generated/GANGeneratorTF/6.txt";
        printMapIndicator(path);
    }
    
}
