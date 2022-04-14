import random
import numpy as np
import ray

class PopulationMARL:
    def __init__(self, population_size, K, T_select,
                 binomial_n, inherit_prob,
                 perturb_prob, perturb_val, hp_range):
        self.population_size = population_size      # num of agents to choose from
        self.K = K      # # step size of Elo rating update given one match result.
        self.T_select = T_select      # agt_j selection threshold
        # inherit variables
        self.binomial_n = binomial_n     # bernoulli is special case of binomial when n=1
        self.inherit_prob = inherit_prob     # hyperparameters are either inherited or not independently with probability 0.5
        # mutation variables
        self.perturb_prob = perturb_prob     # resample_probability
        self.perturb_val = perturb_val      # lower & upper bound for perturbation value
        self.hp_range = hp_range

    def _is_eligible(self, agt_i_key):
        """
        If agt_i completed certain training steps > threshold after
        last evolution, return true.
        """
        return True

    def _is_parent(self, agt_j_key):
        """
        If agt_i completed certain training steps > threshold after
        last evolution, return true.
        """
        return True

    def _s_elo(self, rating_i, rating_j):
        return 1 / (1 + 10**((rating_j - rating_i) / 400))

    def compute_rating(self, prev_rating_i, prev_rating_j, score_i, score_j):
        s = (np.sign(score_i - score_j) + 1) / 2
        s_elo_val = self._s_elo(prev_rating_i, prev_rating_j)
        rating_i = prev_rating_i + self.K * (s - s_elo_val)
        rating_j = prev_rating_j + self.K * (s - s_elo_val)

        return rating_i, rating_j

    def _select_agt_j(self, pol_i_id, population_size, T_select):
        pol_j_id = np.random.randint(low=0, high=population_size, size=None)
        while pol_i_id == pol_j_id:
            pol_j_id = np.random.randint(low=0, high=population_size, size=None)

        agt_i_key = "agt_{}".format(str(pol_i_id))
        agt_j_key = "agt_{}".format(str(pol_j_id))

        pop_actor = ray.util.get_actor("pop_act")
        rating_i = ray.get(pop_actor.get_rating.remote(agt_i_key))
        rating_j = ray.get(pop_actor.get_rating.remote(agt_j_key))

        s_elo_val = self._s_elo(rating_j, rating_i)

        if s_elo_val < T_select:
            return pol_j_id
        else:
            return None

    def _inherit(self, trainer, pol_i_id, pol_j_id):
        pol_i = "ppo_" + str(pol_i_id)
        pol_j = "ppo_" + str(pol_j_id)
        #print("{}_vs_{}".format(pol_i, pol_j))

        # cpy param_j to param_i
        self._cp_weight(trainer, pol_j, pol_i)

        # inherit hyperparam_j to hyperparam_i
        m = np.random.binomial(self.binomial_n, self.inherit_prob, size=1)[0]      # weightage to inherit from agt_i
        return self._inherit_hyperparameters(trainer, pol_j, pol_i, m)

    def _cp_weight(self, trainer, src, dest):
        """
        Copy weights of source policy to destination policy.
        """

        P0key_P1val = {}
        for (k,v), (k2,v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                  trainer.get_policy(src).get_weights().items()):
            P0key_P1val[k] = v2

        trainer.set_weights({dest:P0key_P1val,
                             src:trainer.get_policy(src).get_weights()})

        for (k,v), (k2,v2) in zip(trainer.get_policy(dest).get_weights().items(),
                                  trainer.get_policy(src).get_weights().items()):
            assert (v == v2).all()

    def _inherit_hyperparameters(self, trainer, src, dest, m):
        src_pol = trainer.get_policy(src)
        #print("src_pol.config['lr']", src_pol.config["lr"])

        dest_pol = trainer.get_policy(dest)
        #print("dest_pol.config['lr']", dest_pol.config["lr"])

        dest_pol.config["lr"] = m * dest_pol.config["lr"] + (1-m) * src_pol.config["lr"]
        dest_pol.config["gamma"] = m * dest_pol.config["gamma"] + (1-m) * src_pol.config["gamma"]
        #print("src_pol.config['lr']", src_pol.config["lr"])
        #print("dest_pol.config['lr']", dest_pol.config["lr"])

        return dest_pol

    def _mutate(self, trainer, pol_i_id, pol_i):
        """
        Don't perturb gamma, just resample when applicable.
        """
        if random.random() < self.perturb_prob:
            pol_i.config["lr"] = np.random.uniform(low=self.hp_range["lr"][0], high=self.hp_range["lr"][1], size=None)
            pol_i.config["gamma"] = np.random.uniform(low=self.hp_range["gamma"][1], high=self.hp_range["gamma"][1], size=None)
        elif random.random() < 0.5:
            pol_i.config["lr"] = pol_i.config["lr"] * self.perturb_val[0]
        else: 
            pol_i.config["lr"] = pol_i.config["lr"] * self.perturb_val[1]

        # update hyperparameters in storage
        key = "agt_" + str(pol_i_id)
        pop_actor = ray.util.get_actor("pop_act")
        ray.get(pop_actor.update_hyperparameters.remote(key, pol_i.config["lr"], pol_i.config["gamma"]))
    
        p_id = "ppo_" + str(pol_i_id)
        for w in trainer.workers.remote_workers():
            # Changes in lr can be verifiable in tensorboard or pretty_print.
            new_lr = pol_i.config["lr"]
            w.for_policy.remote(lambda p: p.update_lr_schedule(new_lr), p_id)
            # gamma should be correct but can't verify in tensorboard unless gamma is added to custom metrics in callbacks.
            new_gamma = pol_i.config["gamma"]
            w.for_policy.remote(lambda p: p.update_gamma(new_gamma), p_id)

    def PBT(self, trainer):
        """
        For all agents in population, if agt_i is eligible,
        select agt_j, (i != j), if agt_j is a parent,
        inherit (exploit) & mutate (explore: pertube/resample)
        """
        for i in range(self.population_size):
            pol_i_id = i
            if self._is_eligible(pol_i_id):
                pol_j_id = self._select_agt_j(pol_i_id, self.population_size, self.T_select)
                if pol_j_id is not None:
                    if self._is_parent(pol_j_id):
                        pol_i = self._inherit(trainer, pol_i_id, pol_j_id)
                        self._mutate(trainer, pol_i_id, pol_i)


@ray.remote(num_cpus=0.25, num_gpus=0)
class PopulationActor:
    def __init__(self, population_size, policies):
        self.population_size = population_size
        self.agt_i, self.agt_j = None, None
        self.policies = policies
        self.agt_store = self._create_agt_store(population_size, policies)

    def set_new_agent_pair(self):
        i, j = np.random.randint(low=0, high=self.population_size, size=2)
        while i == j:
            j = np.random.randint(low=0, high=self.population_size, size=None)

        self.agt_i = "agt_" + str(i)
        self.agt_j = "agt_" + str(j)
        print(f"Set agents to {self.agt_i} and {self.agt_j}")

    def get_agent_pair(self):
        return self.agt_i, self.agt_j

    def _create_agt_store(self, population_size, policies):
        """
        Storage for stats of agents in the population.
        """
        store = {}
        for i in range(0, population_size):
            agt_name = "agt_{}".format(str(i))
            store[agt_name] = {"hyperparameters": {"lr":[],
                                                   "gamma":[]},
                               "opponent": [],
                               "score": [],
                               "rating": [],
                               "step": []}

        store = self._init_hyperparameters(store, policies)

        return store

    def _init_hyperparameters(self, store, policies):
        """
        """
        for key, val in store.items():
            _, str_i = key.split("_")
            pol_key = "ppo_" + str_i
            lr = policies[pol_key][3]["lr"]
            gamma = policies[pol_key][3]["gamma"]
            opponent = "NA"
            score = 0
            rating = 0.0
            step = 0

            store[key]["hyperparameters"]["lr"].append(lr)
            store[key]["hyperparameters"]["gamma"].append(gamma)
            store[key]["opponent"].append(opponent)
            store[key]["score"].append(score)
            store[key]["rating"].append(rating)
            store[key]["step"].append(step)

        return store

    def update_hyperparameters(self, key, lr, gamma):
        """
        Note that the hyperparameters are not updated per episode rollout.
        They are only updated after each main training loop when applicable.
        """
        self.agt_store[key]["hyperparameters"]["lr"].append(lr)
        self.agt_store[key]["hyperparameters"]["gamma"].append(gamma)

    def update_rating(self, agt_i_key, agt_j_key, rating_i, rating_j, score_i, score_j):
        self.agt_store[agt_i_key]["opponent"].append(agt_j_key)
        self.agt_store[agt_i_key]["score"].append(score_i)
        self.agt_store[agt_i_key]["rating"].append(rating_i)

        self.agt_store[agt_j_key]["opponent"].append(agt_i_key)
        self.agt_store[agt_j_key]["score"].append(score_j)
        self.agt_store[agt_j_key]["rating"].append(rating_j)

    def get_rating(self, agt_key):
        return self.agt_store[agt_key]["rating"][-1]