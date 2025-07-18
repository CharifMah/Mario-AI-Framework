
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import engine.core.MarioLevelGenerator;

import engine.core.MarioAgent;

public class LevelConfig {
	public final static MarioAgent Agent = new agents.andySloane.Agent();
	
	public final static String LevelPath = "./levels/original/lvl-14.txt";
	
	public final static Boolean isBatchPlay = false;
	
    public final static String LevelFolder = "./levels/original/";
    
    public final static MarioLevelGenerator Generator = new levelGenerators.GAN.LevelGenerator();
    
    public final static int Iteration = 100;
    
    public static String[] getLevelPaths() {
        File folder = new File(LevelFolder);
        File[] files = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".txt"));
        List<String> paths = new ArrayList<>();
        if (files != null) {
            for (File file : files) {
                paths.add(file.getPath());
            }
        }
        return paths.toArray(new String[0]);
    }
}
