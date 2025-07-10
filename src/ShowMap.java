import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

import engine.core.MarioWorld;
import engine.helper.Assets;

public class ShowMap extends JComponent {
    private MarioWorld world;
    private BufferedImage image;
    private float scale;

    public ShowMap(MarioWorld world, float scale) {
        this.world = world;
        this.scale = scale;
        int w = world.level.width;
        int h = world.level.height;
        this.image = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        renderFullMap();
        setPreferredSize(new Dimension((int)(w * scale), (int)(h * scale)));
    }

    public void setScale(float scale) {
        this.scale = scale;
        int w = image.getWidth();
        int h = image.getHeight();
        setPreferredSize(new Dimension((int)(w * scale), (int)(h * scale)));
        revalidate(); // Pour que le JScrollPane s'adapte
        repaint();
    }

    private void renderFullMap() {
        Graphics og = image.getGraphics();
        og.setColor(Color.BLACK);
        og.fillRect(0, 0, image.getWidth(), image.getHeight());
        float oldX = world.cameraX, oldY = world.cameraY;
        for (int y = 0; y < image.getHeight(); y += 240) {
            for (int x = 0; x < image.getWidth(); x += 256) {
                world.cameraX = x;
                world.cameraY = y;
                BufferedImage sub = new BufferedImage(256, 240, BufferedImage.TYPE_INT_ARGB);
                Graphics gsub = sub.getGraphics();
                world.render(gsub);
                og.drawImage(sub, x, y, null);
            }
        }
        world.cameraX = oldX;
        world.cameraY = oldY;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (image != null) {
            g.drawImage(image, 0, 0, (int)(image.getWidth() * scale), (int)(image.getHeight() * scale), null);
        }
    }

    public static void main(String[] args) {
        String levelPath = LevelConfig.LevelPath;
        String levelString = getLevel(levelPath);
        if (levelString == null || levelString.isEmpty()) {
            System.err.println("Niveau vide ou introuvable : " + levelPath);
            return;
        }

        JFrame frame = new JFrame("Mario Map Viewer");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel dummy = new JPanel();
        frame.getContentPane().add(dummy);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        GraphicsConfiguration gc = dummy.getGraphicsConfiguration();
        Assets.init(gc);

        MarioWorld world = new MarioWorld(null);
        world.visuals = true;
        world.initializeVisuals(gc);
        world.initializeLevel(levelString, 400 * 30);

        float initialScale = 2.0f;
        ShowMap showMap = new ShowMap(world, initialScale);

        JScrollPane scrollPane = new JScrollPane(showMap);
        scrollPane.setPreferredSize(new Dimension(1200, 800));

        // Slider de zoom (échelle)
        JSlider slider = new JSlider(5, 40, (int)(initialScale * 10)); // 0.5x à 4.0x
        slider.setMajorTickSpacing(5);
        slider.setMinorTickSpacing(1);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        slider.addChangeListener(e -> {
            float scale = slider.getValue() / 10.0f;
            showMap.setScale(scale);
        });

        JPanel panel = new JPanel(new BorderLayout());
        panel.add(scrollPane, BorderLayout.CENTER);
        panel.add(slider, BorderLayout.SOUTH);

        frame.getContentPane().remove(dummy);
        frame.getContentPane().add(panel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        showMap.repaint();
    }

    private static String getLevel(String filepath) {
        try {
            return new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
            System.err.println("Erreur lors du chargement du niveau : " + filepath);
            return null;
        }
    }
}
