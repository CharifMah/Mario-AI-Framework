import numpy as np
from optimization.algorithms import MapElitesAlgorithm
from optimization.SearchHelper import FeatureMap, Individual
from gan.gan_generator import gan_generate, write_level_txt

# --- PARAMÈTRES ---
LATENT_DIM = 32
MODEL_PATH = "gan_savedmodel"
MAPPING_PATH = "char_mapping.json"
N_LEVELS = 50  # Nombre d'évaluations (modifie pour test rapide)
HEIGHT, WIDTH = 16, 150

# Feature Map : 1D, maximiser nombre de pièces C (adaptable à plusieurs features)
feature_ranges = [(0, WIDTH*HEIGHT)]   # Range pour nombre de pièces
resolutions = [20]                    # 20 bins (grilles) dans la map
feature_map = FeatureMap(feature_ranges, resolutions)

# Logging columns
column_names = ["algo"] + [f"z{i}" for i in range(LATENT_DIM)] + ["fitness", "nb_C"]
bc_names = ["nb_C"]

def count_C(level_txt):
    """Compte le nombre de tuiles 'C' dans le niveau généré."""
    return level_txt.count('C')

def evaluate_latent(latent_vector):
    # Génère le niveau avec le GAN (output txt)
    level_txt = gan_generate(latent_vector, MODEL_PATH, MAPPING_PATH)
    nb_C = count_C(level_txt)
    fitness = nb_C  # Ici on maximise nb de pièces
    features = [nb_C]
    return level_txt, fitness, features

# --- Algorithme MAP-Elites
algo = MapElitesAlgorithm(
    mutation_power=0.5,
    initial_population=50,
    num_to_evaluate=N_LEVELS,
    feature_map=feature_map,
    trial_name="GAN_max_C",
    column_names=column_names,
    bc_names=bc_names,
)

for i in range(N_LEVELS):
    if not algo.is_running():
        break
    ind = algo.generate_individual()
    level_txt, fitness, features = evaluate_latent(ind.param_vector)
    ind.fitness = fitness
    ind.features = features
    ind.statsList = [fitness, features[0]]
    algo.return_evaluated_individual(ind)
    # Enregistre chaque niveau généré pour vérif
    write_level_txt(level_txt, f"../../levels/generated/GANGeneratorTF/{ind.ID}.txt")
    if i % 20 == 0:
        print(f"[{i}] Fitness = {fitness}, nb_C = {features[0]}")

print("Optimisation terminée !")
