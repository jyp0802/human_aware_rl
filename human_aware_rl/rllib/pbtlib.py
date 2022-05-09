import random
import numpy as np
import ray
import itertools

class PopulationMARL:
    def __init__(self, population_size, resample_prob, mutation_factor, hp_to_mutate, hp_range):
        self.population_size = population_size
        self.resample_prob = resample_prob
        self.mutation_factor = mutation_factor
        self.hp_to_mutate = hp_to_mutate
        self.hp_range = hp_range

    def mutate(self, trainer, dest_idx, src_idx):
        dest_pol = "ppo_" + str(dest_idx)
        src_pol = "ppo_" + str(src_idx)

        # self._copy_weights(trainer, src_pol, dest_pol)
        return self._inherit_hp(trainer, src_pol, dest_pol)

    def _clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def _copy_weights(self, trainer, src, dest):
        P0key_P1val = {}
        for (k,v), (k2,v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                  trainer.get_policy(src).get_weights().items()):
            P0key_P1val[k] = v2

        trainer.set_weights({dest:P0key_P1val,
                             src:trainer.get_policy(src).get_weights()})

        for (k,v), (k2,v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                  trainer.get_policy(src).get_weights().items()):
            assert (v == v2).all()

    def _inherit_hp(self, trainer, src, dest):
        src_pol = trainer.get_policy(src)
        dest_pol = trainer.get_policy(dest)

        for hp in self.hp_to_mutate:
            # dest_pol.config[hp] = src_pol.config[hp]
            if random.random() < self.resample_prob:
                if hp in ["lambda", "gamma"]:
                    eps = min(dest_pol.config[hp], 1-dest_pol.config[hp]) / 2
                    dest_pol.config[hp] += [eps, -eps][random.randint(0, 1)]
                else:
                    dest_pol.config[hp] *= self.mutation_factor[random.randint(0, 1)]
                dest_pol.config[hp] = self._clamp(dest_pol.config[hp], *self.hp_range[hp])

        local_worker = trainer.workers.local_worker()
        for hp in self.hp_to_mutate:
            local_worker.for_policy(lambda p: p.update_hp(hp, dest_pol.config[hp]), dest)

        for idx, w in enumerate(trainer.workers.remote_workers()):
            for hp in self.hp_to_mutate:
                new_val = dest_pol.config[hp]
                w.for_policy.remote(lambda p: p.update_hp(hp, new_val), dest)

        return {hp: dest_pol.config[hp] for hp in self.hp_to_mutate}


@ray.remote(num_cpus=0.25, num_gpus=0)
class CombinationPopulationActor:
    def __init__(self, population_size):
        self.population_size = population_size
        self.all_pairs = list(itertools.product(range(population_size), range(population_size)))
        self.reset_agent_pairs()

    def reset_agent_pairs(self):
        self.pairs_to_train = self.all_pairs.copy()
        self.scores = np.zeros((self.population_size, self.population_size))
        self.num_runs = np.zeros((self.population_size, self.population_size))

    def get_agent_pair(self):
        if len(self.pairs_to_train) == 0:
            self.pairs_to_train = self.all_pairs.copy()
        pair_idx = np.random.choice(len(self.pairs_to_train))
        idx_i, idx_j = self.pairs_to_train.pop(pair_idx)
        agt_i, agt_j = f"agt_0_{idx_i}", f"agt_1_{idx_j}"
        return agt_i, agt_j

    def update_scores(self, agt_i, agt_j, score):
        idx_i = int(agt_i.split("_")[-1])
        idx_j = int(agt_j.split("_")[-1])
        self.scores[idx_i, idx_j] += score
        self.num_runs[idx_i, idx_j] += 1

    def get_eval_pairs(self, policies):
        eval_pairs = list(itertools.product(policies, policies))
        eval_pairs = [pair for pair in eval_pairs if "ppo_0" in pair]
        np.random.shuffle(eval_pairs)
        return eval_pairs

    def get_scores(self):
        return self.scores / self.num_runs


@ray.remote(num_cpus=0.25, num_gpus=0)
class NextPopulationActor:
    def __init__(self, population_size):
        self.population_size = population_size
        self.all_pairs = []
        for i in range(population_size):
            self.all_pairs += [(i, i+1), (i, population_size), (i+1, i), (population_size, i)]
        self.all_pairs = list(set(self.all_pairs))
        self.reset_agent_pairs()

    def reset_agent_pairs(self):
        self.pairs_to_train = self.all_pairs.copy()
        self.scores = np.zeros(self.population_size)
        self.num_runs = np.zeros(self.population_size)

    def get_agent_pair(self):
        if len(self.pairs_to_train) == 0:
            self.pairs_to_train = self.all_pairs.copy()
        pair_idx = np.random.choice(len(self.pairs_to_train))
        idx_i, idx_j = self.pairs_to_train.pop(pair_idx)
        agt_i, agt_j = f"agt_0_{idx_i}", f"agt_1_{idx_j}"
        return agt_i, agt_j

    def get_eval_pairs(self, policies):
        eval_pairs = list(itertools.product(policies, policies))
        eval_pairs = [pair for pair in eval_pairs if "ppo_0" in pair]
        np.random.shuffle(eval_pairs)
        return eval_pairs

    def update_scores(self, agt_i, agt_j, score):
        for agt in [agt_i, agt_j]:
            idx = int(agt.split("_")[-1])
            if idx < self.population_size:
                self.scores[idx] += score
                self.num_runs[idx] += 1

    def get_scores(self):
        return self.scores / self.num_runs