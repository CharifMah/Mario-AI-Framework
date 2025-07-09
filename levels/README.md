This folder contains Mario levels. All the folders except original contains generated levels using some of the levels generator that was used in writing "A Comparative Evaluation of Procedural Level Generators in the Mario AI Framework" paper. The original folder contain a modified version of the levels that exists in the VGLC but using a higher resolution.

All generated levels folders have levels from lvl-1.txt to lvl-1000.txt
The original level folder have levels from lvl-1.txt to lvl-15.txt

Generated Levels: http://sokath.com/fdg2014_pcg_evaluation/

VGLC: https://github.com/TheVGLC/TheVGLC

Symbol Reference:
- 'M': Mario Starting Position, not having it will force the engine to start at x = 0 and the first ground floor.
- 'F': Mario finish line, not having it will force the engine to end at x = levelWidth and the first ground floor.
- 'y': Spiky
- 'Y': Winged Spiky
- 'E' or 'g': Goomba
- 'G': Winged Goomba
- 'k': Green Koopa
- 'K': Winged Green Koopa
- 'r': Red Koopa
- 'X': Ground Block
- '#': Pyramind Block
- '%': Jump through platform
- '|': Background for the jump through platform
- '\*': Bullet bill where the top '\*' will be the bullet bill head
- 'B': Bullet bill head
- 'b': Bullet bill neck or body
- '?' or '@': Special Question block
- 'Q' or '!': Coin Question block
- '1': Invisible 1 up block
- '2': Invisible coin bock
- 'D': Used block
- 'S': Normal Brick Block
- 'C': Coing Brick Block
- 'U': Musrhoom Brick Block
- 'L': 1 up Block
- 'o': Coin
- 't': Empty Pipe
- 'T': Pipe with Piranaha Plant in it
- '<': Top left of empty pipe
- '>': Top right of empty pipe
- '[': Left of empty pipe
- ']': Right of empty pipe



Les Générateurs Testés
Notch generator : Générateur par défaut d’Infinite Mario Bros, place les éléments selon des probabilités.

Parameterized Notch : Version paramétrable du précédent (contrôle du nombre d’ennemis, trous, etc.).

Hopper : Génère des niveaux adaptatifs, alternant parties aléatoires et parties pré-conçues.

Launchpad : Générateur basé sur la grammaire et le rythme des niveaux.

ORE (Occupancy-Regulated Extension) : Assemble des petits morceaux de niveaux pré-conçus.

Pattern-based : Utilise l’évolution pour maximiser la présence de certains motifs (patterns) dans le niveau.

Grammatical Evolution (GE) : Utilise l’évolution et des grammaires de design.

Niveaux originaux de SMB : Les vrais niveaux du jeu, pour référence.