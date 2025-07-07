#!/usr/bin/env python3
# models/gan_lsi/run_lsi.py

import os
import argparse
import numpy as np

from .gan.gan_generator import gan_generate, write_level_txt
from .optimization.SearchHelper import FeatureMap, Individual, to_level
from .optimization.algorithms import MapElitesAlgorithm, CMA_ES_Algorithm

def parse_args():
    p = argparse.ArgumentParser(description="Latent Space Illumination")
    p.add_argument("--model-path",
                   type=str,
                   default="models/gan_lsi/checkpoints/gan_final.weights.h5",
                   help="Chemin vers les poids Keras du générateur")
    p.add_argument("--algorithm",
                   choices=["mapelites", "cma_es"],
                   default="mapelites",
                   help="Algorithme QD à utiliser")
    p.add_argument("--evaluations",
                   type=int,
                   default=1000,
                   help="Nombre total d’évaluations")
    p.add_argument("--init-pop",
                   type=int,
                   default=100,
                   help="Pop. initiale (MAP-Elites) / départ (CMA-ES)")
    p.add_argument("--pop-size",
                   type=int,
                   default=20,
                   help="Taille de la population (CMA-ES)")
    p.add_argument("--mutation-power",
                   type=float,
                   default=0.5,
                   help="Écart-type initial de la mutation")
    p.add_argument("--bc1-range",
                   type=float,
                   nargs=2,
                   default=(0.0, 150.0),
                   help="Plage [min max] pour BC1 (n° ennemis)")
    p.add_argument("--bc2-range",
                   type=float,
                   nargs=2,
                   default=(0.0, 25.0),
                   help="Plage [min max] pour BC2 (hauteur moyenne)")
    p.add_argument("--bc1-res",
                   type=int,
                   default=151,
                   help="Résolution de la grille pour BC1")
    p.add_argument("--bc2-res",
                   type=int,
                   default=26,
                   help="Résolution de la grille pour BC2")
    p.add_argument("--java-jar",
                   type=str,
                   default="target/Mario-AI-Framework.jar",
                   help="Chemin vers le jar Java d’évaluation")
    p.add_argument("--java-agent",
                   type=str,
                   default="MarioAI-Astar",
                   help="Nom de l’agent Java pour l’évaluation")
    p.add_argument("--temp-txt",
                   type=str,
                   default="temp_level.txt",
                   help="Fichier temporaire de niveau .txt")
    p.add_argument("--out-archive",
                   type=str,
                   default="models/gan_lsi/results/archive.npy",
                   help="Fichier de sortie pour l’archive MAP-Elites")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_archive), exist_ok=True)

    # 1) Prépare FeatureMap
    feature_ranges = [tuple(args.bc1_range), tuple(args.bc2_range)]
    resolutions    = [args.bc1_res, args.bc2_res]
    fm = FeatureMap(feature_ranges, resolutions)

    # 2) Instancie l’algorithme QD
    column_names = ["method", "fitness", "bc1", "bc2"]
    bc_names     = ["bc1", "bc2"]
    if args.algorithm == "mapelites":
        algo = MapElitesAlgorithm(
            mutation_power     = args.mutation_power,
            initial_population = args.init_pop,
            num_to_evaluate    = args.evaluations,
            feature_map        = fm,
            trial_name         = "run_lsi",
            column_names       = column_names,
            bc_names           = bc_names
        )
    else:
        algo = CMA_ES_Algorithm(
            num_to_evaluate  = args.evaluations,
            mutation_power   = args.mutation_power,
            population_size  = args.pop_size,
            feature_map      = fm,
            trial_name       = "run_lsi",
            column_names     = column_names,
            bc_names         = bc_names
        )

    print(f"▶ Lancement de LSI ({args.algorithm}) pour {args.evaluations} évaluations.")

    # 3) Boucle d’illumination
    while algo.is_running():
        ind = algo.generate_individual()
        z   = ind.param_vector

        # a) Génération & écriture du niveau
        json_lvl = gan_generate(z, args.model_path)
        write_level_txt(json_lvl, args.temp_txt)

        # b) Évaluation via Java CLI
        cmd = (f"java -cp {args.java_jar} GenerateLevel "
               f"--input {args.temp_txt} --agent {args.java_agent}")
        out = os.popen(cmd).read()

        # c) Parsing des BC & du score
        bc1   = float(out.split("Enemies:")[1].split()[0])
        bc2   = float(out.split("AvgHeight:")[1].split()[0])
        score = float(out.split("Score:")[1].split()[0])

        # d) MàJ de l’individu
        ind.features  = (bc1, bc2)
        ind.fitness   = score
        ind.statsList = [score]

        # e) Retour à l’algorithme
        algo.return_evaluated_individual(ind)

    # 4) Sauvegarde finale de l’archive
    fm.save(args.out_archive)
    print(f"✅ Illumination terminée, archive enregistrée dans {args.out_archive}")

if __name__ == "__main__":
    main()
