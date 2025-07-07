import os
import math
import random
import csv
import numpy as np
import pandas as pd
from .SearchHelper import Individual, DecompMatrix, FeatureMap

# Nombre de paramètres latents (à adapter si nécessaire)
num_params = 32
# Fréquence d’enregistrement pour le logging
record_frequency = 20


class CMA_ES_Algorithm:
    def __init__(self,
                 num_to_evaluate,
                 mutation_power,
                 population_size,
                 feature_map: FeatureMap,
                 trial_name: str,
                 column_names,
                 bc_names):
        self.population_size = population_size
        self.num_parents = self.population_size // 2
        self.feature_map = feature_map
        self.all_records = pd.DataFrame(columns=column_names)
        self.sigma = mutation_power
        self.mutation_power = mutation_power
        self.num_to_evaluate = num_to_evaluate
        self.individuals_evaluated = 0
        self.individuals_evaluated_total = 0
        self.trial_name = trial_name
        self.bc_names = bc_names

        self.best = None
        self.mean = np.zeros(num_params)
        self.population = []

        # Setup recombination weights
        self.weights = [math.log(self.num_parents + 0.5) - math.log(i+1)
                        for i in range(self.num_parents)]
        total_weights = sum(self.weights)
        self.weights = np.array([w/total_weights for w in self.weights])
        self.mueff = sum(self.weights)**2 / sum(self.weights**2)

        # Strategy parameters for adaptation
        self.cc = (4 + self.mueff/num_params) / (num_params + 4 + 2*self.mueff/num_params)
        self.cs = (self.mueff + 2) / (num_params + self.mueff + 5)
        self.c1 = 2 / ((num_params + 1.3)**2 + self.mueff)
        self.cmu = min(1-self.c1,
                       2*(self.mueff-2+1/self.mueff)/((num_params+2)**2 + self.mueff))
        self.damps = 1 + 2*max(0,
                               math.sqrt((self.mueff-1)/(num_params+1))-1) + self.cs
        self.chiN = math.sqrt(num_params)*(1 - 1/(4*num_params) + 1/(21*num_params**2))

        # Evolution path
        self.pc = np.zeros(num_params)
        self.ps = np.zeros(num_params)

        # Covariance matrix
        self.C = DecompMatrix(num_params)

    def reset(self):
        """Restart from best or zero."""
        self.mutation_power = self.sigma
        if self.best:
            self.mean = self.best.param_vector.copy()
        else:
            self.mean = np.zeros(num_params)
        self.pc = np.zeros(num_params)
        self.ps = np.zeros(num_params)
        self.C = DecompMatrix(num_params)
        self.individuals_evaluated = 0

    def is_running(self):
        return self.individuals_evaluated_total < self.num_to_evaluate

    def generate_individual(self):
        """Sample a new latent vector."""
        # Sample N(0, sigma) scaled by sqrt(eigenvalues)
        step = np.random.normal(0, self.mutation_power, num_params) * np.sqrt(self.C.eigenvalues)
        step = self.C.eigenbasis.dot(step)
        params = self.mean + step

        ind = Individual()
        ind.param_vector = params
        return ind

    def return_evaluated_individual(self, ind: Individual):
        """Receive back a filled Individual (with fitness, features)."""
        # Assign ID and counters
        ind.ID = self.individuals_evaluated_total
        self.individuals_evaluated += 1
        self.individuals_evaluated_total += 1

        # Add to Map-Elites archive
        self.feature_map.add(ind)

        # Log record
        rec = ["CMA-ES"] + ind.param_vector.tolist() + ind.statsList + list(ind.features)
        self.all_records.loc[ind.ID] = rec

        # Periodic CSV logging
        if self.individuals_evaluated_total % record_frequency == 0:
            elites = list(self.feature_map.elite_map.values())
            if elites:
                os.makedirs("logs", exist_ok=True)
                with open(f"logs/{self.trial_name}_elites_freq{record_frequency}.csv", "a", newline='') as f:
                    writer = csv.writer(f)
                    for e in elites:
                        writer.writerow([e.ID, e.fitness] + list(e.features))

        # Add to current population
        self.population.append(ind)

        # Update best
        if self.best is None or ind.fitness > self.best.fitness:
            self.best = ind

        # If population not full yet, return
        if len(self.population) < self.population_size:
            return

        # Selection of parents
        parents = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.num_parents]
        old_mean = self.mean.copy()
        # Recombination → new mean
        self.mean = sum(p.param_vector * w for p, w in zip(parents, self.weights))

        # Adapt evolution paths
        y = self.mean - old_mean
        z = self.C.invsqrt.dot(y)
        self.ps = (1-self.cs)*self.ps + (math.sqrt(self.cs*(2-self.cs)*self.mueff)/self.mutation_power)*z
        hsig = 1 if (sum(self.ps**2)/num_params /
                     (1-(1-self.cs)**(2*self.individuals_evaluated/self.population_size))
                     < 2 + 4/(num_params+1)) else 0
        self.pc = (1-self.cc)*self.pc + hsig*math.sqrt(self.cc*(2-self.cc)*self.mueff)*y

        # Update covariance
        c1a = self.c1 * (1 - (1-hsig**2)*self.cc*(2-self.cc))
        self.C.C *= (1 - c1a - self.cmu)
        self.C.C += self.c1 * np.outer(self.pc, self.pc)
        for k, w in enumerate(self.weights):
            dv = parents[k].param_vector - old_mean
            self.C.C += w * self.cmu * np.outer(dv, dv) / (self.mutation_power**2)

        # Eigen decomposition update
        self.C.update_eigensystem()

        # Step-size adaptation
        cn = self.cs / self.damps
        ss = sum(self.ps**2)
        self.mutation_power *= math.exp(min(1, cn*(ss/num_params - 1)/2))

        # Restart check
        if self.C.condition_number > 1e14 or self.mutation_power*math.sqrt(max(self.C.eigenvalues))<1e-11:
            self.reset()

        # Clear population for next generation
        self.population.clear()


class MapElitesAlgorithm:
    def __init__(self,
                 mutation_power,
                 initial_population,
                 num_to_evaluate,
                 feature_map: FeatureMap,
                 trial_name: str,
                 column_names,
                 bc_names):
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
        if self.individuals_dispatched < self.initial_population:
            ind.param_vector = np.random.normal(0, 1, num_params)
        else:
            parent = self.feature_map.get_random_elite()
            ind.param_vector = parent.param_vector + \
                               np.random.normal(0, self.mutation_power, num_params)
        self.individuals_dispatched += 1
        return ind

    def return_evaluated_individual(self, ind: Individual):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1

        # Add to archive
        replaced = self.feature_map.add(ind)
        rec = ["MAP-Elites"] + ind.param_vector.tolist() + ind.statsList + list(ind.features)
        self.all_records.loc[ind.ID] = rec

        # Periodic log
        if self.individuals_evaluated % record_frequency == 0:
            os.makedirs("logs", exist_ok=True)
            elites = list(self.feature_map.elite_map.values())
            with open(f"logs/{self.trial_name}_elites_freq{record_frequency}.csv", "a", newline='') as f:
                writer = csv.writer(f)
                for e in elites:
                    writer.writerow([e.ID, e.fitness] + list(e.features))
