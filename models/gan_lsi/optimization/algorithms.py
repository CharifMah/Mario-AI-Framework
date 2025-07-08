import os
import csv
import numpy as np
import pandas as pd
from .SearchHelper import Individual, FeatureMap

LATENT_DIM = 32
record_frequency = 20

class MapElitesAlgorithm:
    def __init__(self, mutation_power, initial_population, num_to_evaluate, feature_map, trial_name, column_names, bc_names):
        self.mutation_power = mutation_power
        self.initial_population = initial_population
        self.num_to_evaluate = num_to_evaluate
        self.feature_map = feature_map
        self.trial_name = trial_name
        self.all_records = pd.DataFrame(columns=column_names)
        self.bc_names = bc_names
        self.individuals_evaluated = 0
        self.individuals_dispatched = 0

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        ind = Individual()
        if self.individuals_dispatched < self.initial_population or not self.feature_map.elite_indices:
            ind.param_vector = np.random.normal(0, 1, LATENT_DIM)
        else:
            parent = self.feature_map.get_random_elite()
            ind.param_vector = parent.param_vector + np.random.normal(0, self.mutation_power, LATENT_DIM)
        self.individuals_dispatched += 1
        return ind

    def return_evaluated_individual(self, ind: Individual):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1
        replaced = self.feature_map.add(ind)
        rec = ["MAP-Elites"] + ind.param_vector.tolist() + ind.statsList
        self.all_records.loc[ind.ID] = rec
        # Logging (optionnel)
        if self.individuals_evaluated % record_frequency == 0:
            os.makedirs("logs", exist_ok=True)
            elites = list(self.feature_map.elite_map.values())
            with open(f"logs/{self.trial_name}_elites_freq{record_frequency}.csv", "a", newline='') as f:
                writer = csv.writer(f)
                for e in elites:
                    writer.writerow([e.ID, e.fitness] + list(e.features))
