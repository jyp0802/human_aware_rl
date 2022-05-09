from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, EVENT_TYPES
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import Agent, AgentPair
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.agents.ppo.ppo_tf_policy import ppo_surrogate_loss, kl_and_loss_stats, vf_preds_fetches, postprocess_ppo_gae, clip_gradients, setup_config
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.schedules import ConstantSchedule
from human_aware_rl.rllib.utils import softmax, get_base_ae, get_required_arguments, iterable_equal
from human_aware_rl.rllib.pbtlib import CombinationPopulationActor, NextPopulationActor, PopulationMARL
from datetime import datetime
import tempfile
import gym
import numpy as np
import os, copy, dill
import ray
import logging
import itertools

action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
obs_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")


class RlLibAgent(Agent):
    """ 
    Class for wrapping a trained RLLib Policy object into an Overcooked compatible Agent
    """
    def __init__(self, policy, agent_index, featurize_fn):
        self.policy = policy
        self.agent_index = agent_index
        self.featurize = featurize_fn

    def reset(self):
        # Get initial rnn states and add batch dimension to each
        if hasattr(self.policy.model, 'get_initial_state'):
            self.rnn_state = [np.expand_dims(state, axis=0) for state in self.policy.model.get_initial_state()]
        elif hasattr(self.policy, "get_initial_state"):
            self.rnn_state = [np.expand_dims(state, axis=0) for state in self.policy.get_initial_state()]
        else:
            self.rnn_state = []

    def action_probabilities(self, state):
        """
        Arguments:
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns:
            - Normalized action probabilities determined by self.policy
        """
        # Preprocess the environment state
        obs = self.featurize(state, debug=False)
        my_obs = obs[self.agent_index]

        # Compute non-normalized log probabilities from the underlying model
        logits = self.policy.compute_actions(np.array([my_obs]), self.rnn_state)[2]['action_dist_inputs']

        # Softmax in numpy to convert logits to normalized probabilities
        return softmax(logits)

    def action(self, state):
        """
        Arguments: 
            - state (Overcooked_mdp.OvercookedState) object encoding the global view of the environment
        returns: 
            - the argmax action for a single observation state
            - action_info (dict) that stores action probabilities under 'action_probs' key
        """
        # Preprocess the environment state
        obs = self.featurize(state)
        my_obs = obs[self.agent_index]

        # Use Rllib.Policy class to compute action argmax and action probabilities
        [action_idx], rnn_state, info = self.policy.compute_actions(np.array([my_obs]), self.rnn_state)
        agent_action =  Action.INDEX_TO_ACTION[action_idx]
        
        # Softmax in numpy to convert logits to normalized probabilities
        logits = info['action_dist_inputs']
        action_probabilities = softmax(logits)

        agent_action_info = {'action_probs' : action_probabilities}
        self.rnn_state = rnn_state

        return agent_action, agent_action_info


class OvercookedPopulationMultiAgent(MultiAgentEnv):
    """
    Class used to wrap OvercookedEnv in an Rllib compatible multi-agent environment
    """

    # Default environment params used for creation
    DEFAULT_CONFIG = {
        # To be passed into OvercookedGridWorld constructor
        "mdp_params" : {
            "layout_name" : "cramped_room",
            "shaped_reward_params" : {}
        },
        # To be passed into OvercookedEnv constructor
        "env_params" : {
            "horizon" : 400
        },
        # To be passed into OvercookedPopulationMultiAgent constructor
        "multi_agent_params" : {
            "reward_shaping_factor" : 0.0,
            "reward_shaping_horizon" : 0,
            "use_phi" : True
        }
    }

    def __init__(self, base_env, reward_shaping_factor=0.0, reward_shaping_horizon=0,
                            use_phi=True):
        """
        base_env: OvercookedEnv
        reward_shaping_factor (float): Coefficient multiplied by dense reward before adding to sparse reward to determine shaped reward
        reward_shaping_horizon (int): Timestep by which the reward_shaping_factor reaches zero through linear annealing
        use_phi (bool): Whether to use 'shaped_r_by_agent' or 'phi_s_prime' - 'phi_s' to determine dense reward
        """
        self.base_env = base_env
        # since we are not passing featurize_fn in as an argument, we create it here and check its validity
        self.featurize_fn_map = lambda state: self.base_env.lossless_state_encoding_mdp(state)
        self._initial_reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_factor = reward_shaping_factor
        self.reward_shaping_horizon = reward_shaping_horizon
        self.use_phi = use_phi
        self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        # self.reset()

    def _setup_observation_space(self):
        dummy_state = self.base_env.mdp.get_standard_start_state()

        featurize_fn_ppo = lambda state: self.base_env.lossless_state_encoding_mdp(state)
        obs_shape = featurize_fn_ppo(dummy_state)[0].shape
        high = np.ones(obs_shape) * float("inf")
        low = np.ones(obs_shape) * 0
        self.ppo_observation_space = gym.spaces.Box(np.float64(low), np.float64(high), dtype=np.float64)

    def _get_featurize_fn(self, agent_id):
        return lambda state: self.base_env.lossless_state_encoding_mdp(state)

    def _get_obs(self, state):
        ob_p0 = self._get_featurize_fn(self.curr_agents[0])(state)[0]
        ob_p1 = self._get_featurize_fn(self.curr_agents[1])(state)[1]

        return ob_p0, ob_p1

    def _populate_agents(self):
        pop_actor = ray.util.get_actor("pop_act")
        agents = ray.get(pop_actor.get_agent_pair.remote())
        return agents

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v


    def step(self, action_dict):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
        
        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        action = [action_dict[self.curr_agents[0]], action_dict[self.curr_agents[1]]]
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid"%(action, type(action))
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        # take a step in the current base environment

        if self.use_phi:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=True)
            potential = info['phi_s_prime'] - info['phi_s']
            dense_reward = (potential, potential)
        else:
            next_state, sparse_reward, done, info = self.base_env.step(joint_action, display_phi=False)
            dense_reward = info["shaped_r_by_agent"]

        ob_p0, ob_p1 = self._get_obs(next_state)

        shaped_reward_p0 = sparse_reward + self.reward_shaping_factor * dense_reward[0]
        shaped_reward_p1 = sparse_reward + self.reward_shaping_factor * dense_reward[1]
        
        obs = { self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1 }
        rewards = { self.curr_agents[0]: shaped_reward_p0, self.curr_agents[1]: shaped_reward_p1 }
        dones = { self.curr_agents[0]: done, self.curr_agents[1]: done, "__all__": done }
        infos = { self.curr_agents[0]: info, self.curr_agents[1]: info }
        return obs, rewards, dones, infos

    def reset(self, regen_mdp=True):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to 
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset(regen_mdp)
        self.curr_agents = self._populate_agents()
        ob_p0, ob_p1 = self._get_obs(self.base_env.state)
        return {self.curr_agents[0]: ob_p0, self.curr_agents[1]: ob_p1}
    
    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(self._initial_reward_shaping_factor, timesteps, self.reward_shaping_horizon)
        self.set_reward_shaping_factor(new_factor)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor

    def seed(self, seed):
        """
        set global random seed to make environment deterministic
        """
        # Our environment is already deterministic
        pass
    
    @classmethod
    def from_config(cls, env_config):
        """
        Factory method for generating environments in style with rllib guidlines

        env_config (dict):  Must contain keys 'mdp_params', 'env_params' and 'multi_agent_params', the last of which
                            gets fed into the OvercookedPopulationMultiAgent constuctor

        Returns:
            OvercookedPopulationMultiAgent instance specified by env_config params
        """
        assert env_config and "env_params" in env_config and "multi_agent_params" in env_config
        assert "mdp_params" in env_config or "mdp_params_schedule_fn" in env_config, \
            "either a fixed set of mdp params or a schedule function needs to be given"
        # "layout_name" and "shaped_reward_params"
        if "mdp_params" in env_config:
            mdp_params = env_config["mdp_params"]
            outer_shape = None
            mdp_params_schedule_fn = None
        elif "mdp_params_schedule_fn" in env_config:
            mdp_params = None
            outer_shape = env_config["outer_shape"]
            mdp_params_schedule_fn = env_config["mdp_params_schedule_fn"]

        # "start_state_fn" and "horizon"
        env_params = env_config["env_params"]
        # "reward_shaping_factor"
        multi_agent_params = env_config["multi_agent_params"]
        base_ae = get_base_ae(mdp_params, env_params, outer_shape, mdp_params_schedule_fn)
        base_env = base_ae.env

        return cls(base_env, **multi_agent_params)



##################
# Training Utils #
##################

def get_population_rllib_eval_function(eval_params, eval_mdp_params, env_params, outer_shape, verbose=False):
    """
    Used to "curry" rllib evaluation function by wrapping additional parameters needed in a local scope, and returning a
    function with rllib custom_evaluation_function compatible signature

    eval_params (dict): Contains 'num_games' (int), 'display' (bool), and 'ep_length' (int)
    mdp_params (dict): Used to create underlying OvercookedMDP (see that class for configuration)
    env_params (dict): Used to create underlying OvercookedEnv (see that class for configuration)
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    Note: Agent policies are shuffled each time, so agent_0_policy_str and agent_1_policy_str are symmetric
    Returns:
        _evaluate (func): Runs an evaluation specified by the curried params, ignores the rllib parameter 'evaluation_workers'
    """

    def _evaluate(trainer, evaluation_workers):
        if verbose:
            print("Computing rollout of current trained policy")

        # Randomize starting indices
        policies = evaluation_workers.foreach_policy(lambda x, y: y)
        
        pop_actor = ray.util.get_actor("pop_act")
        eval_pairs = ray.get(pop_actor.get_eval_pairs.remote(policies))

        ep_returns = []
        for (agent_0_policy, agent_1_policy) in eval_pairs:
            print(f"Evaluating with agents {agent_0_policy} and {agent_1_policy}")

            # Get the corresponding rllib policy objects for each policy string name
            agent_0_policy = trainer.get_policy(agent_0_policy)
            agent_1_policy = trainer.get_policy(agent_1_policy)

            # Compute the evauation rollout. Note this doesn't use the rllib passed in evaluation_workers, so this 
            # computation all happens on the CPU. Could change this if evaluation becomes a bottleneck
            results = evaluate(eval_params, eval_mdp_params, outer_shape, agent_0_policy, agent_1_policy, verbose=verbose)

            ep_returns.append(np.mean(results['ep_returns']))
        # Log any metrics we care about for rllib tensorboard visualization
        metrics = {}
        metrics['average_sparse_reward'] = np.mean(ep_returns)
        return metrics

    return _evaluate


def evaluate(eval_params, mdp_params, outer_shape, agent_0_policy, agent_1_policy, agent_0_featurize_fn=None, agent_1_featurize_fn=None, verbose=False):
    """
    Used to visualize rollouts of trained policies

    eval_params (dict): Contains configurations such as the rollout length, number of games, and whether to display rollouts
    mdp_params (dict): OvercookedMDP compatible configuration used to create environment used for evaluation
    outer_shape (list): a list of 2 item specifying the outer shape of the evaluation layout
    agent_0_policy (rllib.Policy): Policy instance used to map states to action logits for agent 0
    agent_1_policy (rllib.Policy): Policy instance used to map states to action logits for agent 1
    agent_0_featurize_fn (func): Used to preprocess states for agent 0, defaults to lossless_state_encoding if 'None'
    agent_1_featurize_fn (func): Used to preprocess states for agent 1, defaults to lossless_state_encoding if 'None'
    """
    # if verbose:
    #     print("eval mdp params", mdp_params)
    evaluator = get_base_ae(mdp_params, {"horizon" : eval_params['ep_length'], "num_mdp":1}, outer_shape)

    # Override pre-processing functions with defaults if necessary
    agent_0_featurize_fn = agent_0_featurize_fn if agent_0_featurize_fn else evaluator.env.lossless_state_encoding_mdp
    agent_1_featurize_fn = agent_1_featurize_fn if agent_1_featurize_fn else evaluator.env.lossless_state_encoding_mdp

    # Wrap rllib policies in overcooked agents to be compatible with Evaluator code
    agent0 = RlLibAgent(agent_0_policy, agent_index=0, featurize_fn=agent_0_featurize_fn)
    agent1 = RlLibAgent(agent_1_policy, agent_index=1, featurize_fn=agent_1_featurize_fn)

    # Compute rollouts
    if 'store_dir' not in eval_params:
        eval_params['store_dir'] = None
    if 'display_phi' not in eval_params:
        eval_params['display_phi'] = False
    results = evaluator.evaluate_agent_pair(AgentPair(agent0, agent1),
                                            num_games=eval_params['num_games'],
                                            display=eval_params['display'],
                                            dir=eval_params['store_dir'],
                                            display_phi=eval_params['display_phi'],
                                            info=verbose)

    return results


###########################
# rllib.Trainer functions #
###########################

def gen_population_trainer_from_params(params):
    # All ray environment set-up
    if 'verbose' not in params:
        params['verbose'] = False
    if not ray.is_initialized():
        init_params = {
            "ignore_reinit_error" : True,
            "include_webui" : False,
            "temp_dir" : params['ray_params']['temp_dir'],
            "log_to_driver" : params['verbose'],
            "logging_level" : logging.INFO if params['verbose'] else logging.CRITICAL
        }
        ray.init(**init_params)
    register_env("overcooked_multi_agent", params['ray_params']['env_creator'])
    ModelCatalog.register_custom_model(params['ray_params']['custom_model_id'], params['ray_params']['custom_model_cls'])

    # Parse params
    model_params = params['model_params']
    training_params = params['training_params']
    environment_params = params['environment_params']
    population_params = params['population_params']
    evaluation_params = params['evaluation_params']
    multi_agent_params = params['environment_params']['multi_agent_params']

    # Rllib compatible way of setting the directory we store agent checkpoints in
    logdir_prefix = "{0}_{1}_{2}".format(params["experiment_name"], params['training_params']['seed'], timestr)
    def custom_logger_creator(config):
        """Creates a Unified logger that stores results in <params['results_dir']>/<params["experiment_name"]>_<seed>_<timestamp>
        """
        results_dir = params['results_dir']
        if not os.path.exists(results_dir):
            try:
                os.makedirs(results_dir)
            except Exception as e:
                print("error creating custom logging dir. Falling back to default logdir {}".format(DEFAULT_RESULTS_DIR))
                results_dir = DEFAULT_RESULTS_DIR
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=results_dir)
        logger = UnifiedLogger(config, logdir, loggers=None)
        return logger

    env = OvercookedPopulationMultiAgent.from_config(environment_params)

    # Returns a properly formatted policy tuple to be passed into ppotrainer config
    def gen_policy():
        config = {
            "model" : {
                "custom_options" : model_params,
                "custom_model" : "MyPPOModel"
            }
        }
        for hp in population_params["hp_to_mutate"]:
            config[hp] = np.random.uniform(
                low=population_params["hp_range"][hp][0],
                high=population_params["hp_range"][hp][1],
                size=None)
        return (None, env.ppo_observation_space, env.action_space, config)

    if "outer_shape" not in environment_params:
        environment_params["outer_shape"] = None

    if "mdp_params" in environment_params:
        environment_params["eval_mdp_params"] = environment_params["mdp_params"]

    _custom_eval_function = get_population_rllib_eval_function(
        evaluation_params, environment_params['eval_mdp_params'], environment_params['env_params'],
        environment_params["outer_shape"], verbose=params['verbose'])

    # Create rllib compatible multi-agent config based on params
    _multiagent = {}
    if population_params["train_type"] == "combination":
        def policy_mapping_fn(agent_id):
            _, _, i = agent_id.split("_")
            return f"ppo_{i}"

        _multiagent["policies"] = {f"ppo_{i}" : gen_policy() for i in range(population_params["population_size"])}
        _multiagent["policy_mapping_fn"] = policy_mapping_fn
        _multiagent["policies_to_train"] = [f"ppo_{i}" for i in range(population_params["population_size"])]

        population_actor = CombinationPopulationActor.options(name="pop_act").remote(
            population_params["population_size"])

    elif population_params["train_type"] == "next":
        policies = {f"ppo_{i}" : gen_policy() for i in range(population_params["population_size"])}
        policies["random"] = gen_policy()

        def policy_mapping_fn(agent_id):
            _, _, i = agent_id.split("_")
            if int(i) < population_params["population_size"]:
                return f"ppo_{i}"
            return "random"

        _multiagent["policies"] = policies
        _multiagent["policy_mapping_fn"] = policy_mapping_fn
        _multiagent["policies_to_train"] = [f"ppo_{i}" for i in range(population_params["population_size"])]

        population_actor = NextPopulationActor.options(name="pop_act").remote(
            population_params["population_size"])

    population_marl = PopulationMARL(
        population_params["population_size"], population_params["resample_prob"], population_params["mutation_factor"],
        population_params["hp_to_mutate"], population_params["hp_range"])

    class PopulationTrainingCallbacks(DefaultCallbacks):

        def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
            pass

        def on_episode_step(self, worker, base_env, episode, **kwargs):
            pass

        def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
            # Get rllib.OvercookedMultiAgentEnv refernce from rllib wraper
            env = base_env.get_unwrapped()[0]
            # print(env.curr_agents)
            # Both agents share the same info so it doesn't matter whose we use, just use 0th agent's
            info_dict = episode.last_info_for(env.curr_agents[0])
            ep_info = info_dict["episode"]

            # Parse info dicts generated by OvercookedEnv
            tot_sparse_reward = ep_info["ep_sparse_r"]
            tot_shaped_reward = ep_info["ep_shaped_r"]

            # Store metrics where they will be visible to rllib for tensorboard logging
            episode.custom_metrics["sparse_reward"] = tot_sparse_reward
            episode.custom_metrics["shaped_reward"] = tot_shaped_reward

            agents = []
            scores = []
            for k, v in episode.agent_rewards.items():
                agents.append(k[0])
                scores.append(v)

            pop_actor = ray.util.get_actor("pop_act")
            ray.get(pop_actor.update_scores.remote(agents[0], agents[1], scores[0]))

        def on_sample_end(self, worker, samples, **kwargs):
            pass

        def on_train_result(self, trainer, result, **kwargs):
            # Anneal the reward shaping coefficient based on environment paremeters and current timestep
            timestep = result['timesteps_total']
            trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: env.anneal_reward_shaping_factor(timestep)))

            pop_actor = ray.util.get_actor("pop_act")
            all_scores = ray.get(pop_actor.get_scores.remote())

            if population_params["train_type"] == "combination":
                mean_scores = np.mean(all_scores, axis=1)
                worst_agt_idx = np.argmin(mean_scores)
                best_agt_idx = np.argmax(mean_scores)
            elif population_params["train_type"] == "next":
                mean_scores = all_scores
                worst_agt_idx = np.argmin(all_scores)
                best_agt_idx = np.argmax(all_scores)

            for agt_idx in range(population_params["population_size"]):
                result[f"agt_{agt_idx}_score"] = mean_scores[agt_idx]

            new_hp = population_marl.mutate(trainer, worst_agt_idx, best_agt_idx)
            for hp in population_params["hp_to_mutate"]:
                result[f"agt_{worst_agt_idx}_{hp}"] = new_hp[hp]

            ray.get(pop_actor.reset_agent_pairs.remote())

        def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
            pass
    
    class PopulationMixin:
        def __init__(self, config):
            self.config = config

        def update_hp(self, hp, val):
            self.config[hp] = val
            if hp == "lr":
                self.lr_schedule = ConstantSchedule(val, framework=None)

    def setup_mixins(policy, obs_space, action_space, config):
        ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
        KLCoeffMixin.__init__(policy, config)
        EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                    config["entropy_coeff_schedule"])
        LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
        PopulationMixin.__init__(policy, config)

    PopulationPolicy = PPOTFPolicy.with_updates(
        name="PopulationPolicy",
        before_loss_init=setup_mixins,
        mixins=[
            LearningRateSchedule,
            EntropyCoeffSchedule,
            KLCoeffMixin,
            ValueNetworkMixin,
            PopulationMixin,
        ])

    def get_policy_class(config):
        return PopulationPolicy

    PopulationTrainer = PPOTrainer.with_updates(
        name="PopulationTrainer",
        default_policy=PopulationPolicy,
        get_policy_class=get_policy_class
        )

    trainer = PopulationTrainer(
        env="overcooked_multi_agent", 
        config={
            "multiagent": _multiagent,
            "callbacks" : PopulationTrainingCallbacks,
            "custom_eval_function" : _custom_eval_function,
            "env_config" : environment_params,
            "eager" : False,
            **training_params
        },
        logger_creator=custom_logger_creator)

    return trainer


### Serialization ###

def save_trainer(trainer, params, path=None):
    """
    Saves a serialized trainer checkpoint at `path`. If none provided, the default path is
    ~/ray_results/<experiment_results_dir>/checkpoint_<i>/checkpoint-<i>

    Note that `params` should follow the same schema as the dict passed into `gen_population_trainer_from_params`
    """
    # Save trainer
    save_path = trainer.save(path)

    # Save params used to create trainer in /path/to/checkpoint_dir/config.pkl
    config = copy.deepcopy(params)
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")

    # Note that we use dill (not pickle) here because it supports function serialization
    with open(config_path, "wb") as f:
        dill.dump(config, f)
    return save_path

def load_trainer(save_path):
    """
    Returns a ray compatible trainer object that was previously saved at `save_path` by a call to `save_trainer`
    Note that `save_path` is the full path to the checkpoint FILE, not the checkpoint directory
    """
    # Read in params used to create trainer
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")
    with open(config_path, "rb") as f:
        # We use dill (instead of pickle) here because we must deserialize functions
        config = dill.load(f)
    
    # Override this param to lower overhead in trainer creation
    config['training_params']['num_workers'] = 0

    # Get un-trained trainer object with proper config
    trainer = gen_population_trainer_from_params(config)

    # Load weights into dummy object
    trainer.restore(save_path)
    return trainer

def get_agent_from_trainer(trainer, policy_id="ppo", agent_index=0):
    policy = trainer.get_policy(policy_id)
    dummy_env = trainer.env_creator(trainer.config['env_config'])
    featurize_fn = dummy_env.featurize_fn_map
    agent = RlLibAgent(policy, agent_index, featurize_fn=featurize_fn)
    return agent

def get_agent_pair_from_trainer(trainer, policy_id_0='ppo', policy_id_1='ppo'):
    agent0 = get_agent_from_trainer(trainer, policy_id=policy_id_0)
    agent1 = get_agent_from_trainer(trainer, policy_id=policy_id_1)
    return AgentPair(agent0, agent1)


def load_agent_pair(save_path, policy_id_0='ppo', policy_id_1='ppo'):
    """
    Returns an Overcooked AgentPair object that has as player 0 and player 1 policies with 
    ID policy_id_0 and policy_id_1, respectively
    """
    trainer = load_trainer(save_path)
    return get_agent_pair_from_trainer(trainer, policy_id_0, policy_id_1)

def load_agent(save_path, policy_id='ppo', agent_index=0):
    """
    Returns an RllibAgent (compatible with the Overcooked Agent API) from the `save_path` to a previously
    serialized trainer object created with `save_trainer`

    The trainer can have multiple independent policies, so extract the one with ID `policy_id` to wrap in
    an RllibAgent

    Agent index indicates whether the agent is player zero or player one (or player n in the general case)
    as the featurization is not symmetric for both players
    """
    trainer = load_trainer(save_path)
    return get_agent_from_trainer(trainer, policy_id=policy_id, agent_index=agent_index)


