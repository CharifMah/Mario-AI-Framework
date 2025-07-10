import engine.core.MarioLevelModel;
import engine.sprites.Mario;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;
import java.util.*;

public class MarioLevelIndicators {

    // --- Métriques ---

    // Leniency : score basé sur ennemis et obstacles
    public static double computeLeniency(MarioLevelModel level) {
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
        int width = level.getWidth();
        int height = level.getHeight();

        for (int y = 0; y < height; y++) {
            boolean inSeg = false;
            for (int x = 0; x < width; x++) {
                char c = level.getBlock(x, y);
                if (isGround(c)) {
                    if (!inSeg) {
                        seg++;
                        inSeg = true;
                    }
                } else {
                    inSeg = false;
                }
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
    
	// cm - Compte les groupes de colonnes consécutives sans aucun bloc de sol
    public static List<int[]> findVerticalHoles(MarioLevelModel level) {
        int width = level.getWidth();
        int height = level.getHeight();
        List<int[]> holes = new ArrayList<>();
        boolean inHole = false;
        int holeStart = -1;

        for (int x = 0; x < width; x++) {
            boolean columnIsEmpty = true;
            for (int y = 0; y < height; y++) {
                if (isGround(level.getBlock(x, y))) {
                    columnIsEmpty = false;
                    break;
                }
            }
            if (columnIsEmpty && !inHole) {
                inHole = true;
                holeStart = x;
            } else if (!columnIsEmpty && inHole) {
                inHole = false;
                holes.add(new int[]{holeStart, x}); // [début inclus, fin exclu]
            }
        }
        // Si le trou va jusqu'à la dernière colonne
        if (inHole) {
            holes.add(new int[]{holeStart, width});
        }
        return holes;
    }
    
    public static int countTotalVerticalHoles(MarioLevelModel level) {
    	List<int[]> holes = findVerticalHoles(level);
        int count = 0;
        for (int[] hole : holes) {
            int width = hole[1] - hole[0];
            if (width > 6) {
                System.out.println("Trou vertical de x=" + hole[0] + " à x=" + (hole[1] - 1) + " (largeur=" + width + ")");
            }
            count++;
        }
        return count;
    }

    // Compte le nombre de trous injumpables selon la distance et la hauteur de saut max
    public static int countUnjumpableHoles(MarioLevelModel level, int maxJumpDistance, int maxJumpHeight) {
        int width = level.getWidth();
        int height = level.getHeight();
        int count = 0;

        // Recherche la ligne du sol principal (la plus basse contenant du sol)
        int groundLevel = -1;
        for (int y = height - 1; y >= 0; y--) {
            for (int x = 0; x < width; x++) {
                if (isGround(level.getBlock(x, y))) {
                    groundLevel = y;
                    break;
                }
            }
            if (groundLevel != -1) break;
        }
        if (groundLevel == -1) return 0; // Pas de sol trouvé

        int x = 0;
        while (x < width) {
            // Cherche le début d'un trou sur la ligne du sol principal
            while (x < width && isGround(level.getBlock(x, groundLevel))) {
                x++;
            }
            if (x >= width) break; // Plus de trous à analyser

            int holeStart = x;

            // Trouve la fin du trou
            while (x < width && !isGround(level.getBlock(x, groundLevel))) {
                x++;
            }
            int holeEnd = x;
            int holeWidth = holeEnd - holeStart;

            // Vérifie si le trou est trop large pour être sauté
            if (holeWidth > maxJumpDistance) {
                boolean platformFound = false;

                // Recherche de plateformes au-dessus du trou, à une hauteur accessible
                for (int dy = 1; dy <= maxJumpHeight && !platformFound; dy++) {
                    int platY = groundLevel - dy;
                    if (platY < 0) break;

                    int tx = holeStart;
                    while (tx < holeEnd && !platformFound) {
                        // Cherche le début d'une plateforme
                        if (isGround(level.getBlock(tx, platY))) {
                            int platformStart = tx;
                            int platformLength = 0;
                            // Cherche la longueur de la plateforme continue
                            while (tx < holeEnd && isGround(level.getBlock(tx, platY))) {
                                platformLength++;
                                tx++;
                            }
                            int platformEnd = platformStart + platformLength;

                            // Vérifie si Mario peut atteindre cette plateforme depuis les bords du trou
                            // Depuis le bord gauche du trou
                            int leftEdge = holeStart - 1;
                            if (leftEdge >= 0) {
                                int horizontalDistance = platformStart - leftEdge;
                                if (horizontalDistance >= 0 && horizontalDistance <= maxJumpDistance) {
                                    platformFound = true;
                                }
                                int distanceToEnd = platformEnd - 1 - leftEdge;
                                if (distanceToEnd >= 0 && distanceToEnd <= maxJumpDistance) {
                                    platformFound = true;
                                }
                            }
                            // Depuis le bord droit du trou
                            int rightEdge = holeEnd;
                            if (rightEdge < width) {
                                int horizontalDistance = rightEdge - platformEnd + 1;
                                if (horizontalDistance >= 0 && horizontalDistance <= maxJumpDistance) {
                                    platformFound = true;
                                }
                                int distanceToStart = rightEdge - platformStart;
                                if (distanceToStart >= 0 && distanceToStart <= maxJumpDistance) {
                                    platformFound = true;
                                }
                            }
                        } else {
                            tx++;
                        }
                    }
                }

                if (!platformFound) {
                    System.out.println("UNJUMPABLE HOLE found from x=" + holeStart + " to x=" + holeEnd + " (width=" + holeWidth + ")");
                    count++;
                }
            }
        }
        return count;
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
        System.out.println("countUnjumpableHoles : " + countUnjumpableHoles(level,6,6));
        System.out.println("Liste des trous de 6 de largeur ou plus : ");
        System.out.println("countTotalVerticalHoles : " + countTotalVerticalHoles(level));
    }

    // --- Exemple d'utilisation ---
    public static void main(String[] args) {
        printMapIndicator(LevelConfig.LevelPath);
    }
    
}
