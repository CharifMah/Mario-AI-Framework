
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import engine.core.MarioAgent;

public class LevelConfig {
	public final static MarioAgent Agent = new agents.robinBaumgarten.Agent();
	
	public final static String LevelPath = "./levels/generated/GANGeneratorTF/3.txt";
	
	public final static Boolean isBatchPlay = true;
	
    public final static String LevelFolder = "./levels/generated/GANGeneratorTF/";
    
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
