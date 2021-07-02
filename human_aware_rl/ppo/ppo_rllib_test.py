import unittest, os, shutil, pickle, ray, random, argparse, sys, glob
os.environ['RUN_ENV'] = 'local'
from human_aware_rl.ppo.ppo_rllib_client import ex
from human_aware_rl.ppo.ppo_rllib_from_params_client import ex_fp
from human_aware_rl.static import PPO_EXPECTED_DATA_PATH
from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.rllib.rllib import PPOAgent
from human_aware_rl.imitation.behavior_cloning_tf2 import train_bc_model, get_bc_params
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair
import tensorflow as tf
import numpy as np

# Note: using the same seed across architectures can still result in differing values
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

class TestPPORllib(unittest.TestCase):

    """
    Unittests for rllib PPO training loop

    compute_pickle (bool):      Whether the results of this test should be stored as the expected values for future tests
    strict (bool):              Whether the results of this test should be compared against expected values for exact match
    check_performance (bool):   Whether to require training to achieve `min_performance` reward in order to count as "success"
    min_performance (int):      Minimum sparse reward that must be achieved during training for test to count as "success"

    Note, this test always performs a basic sanity check to verify some learning is happening, even if the `strict` param is false
    """

    def __init__(self, test_name, compute_pickle, strict, check_performance, min_performance):
        super(TestPPORllib, self).__init__(test_name)
        self.compute_pickle = compute_pickle
        self.strict = strict
        self.check_performance = check_performance
        self.min_performance = min_performance
    
    def setUp(self):
        set_global_seed(0)

        # Temporary disk space to store logging results from tests
        self.temp_results_dir = os.path.join(os.path.abspath('.'), 'results_temp')
        self.temp_model_dir = os.path.join(os.path.abspath('.'), 'model_temp')
        self.temp_agent_dir = os.path.join(os.path.abspath('.'), 'agent_temp')


        # Make all necessary directories
        if not os.path.exists(self.temp_model_dir):
            os.makedirs(self.temp_model_dir)

        if not os.path.exists(self.temp_results_dir):
            os.makedirs(self.temp_results_dir)

        # Load in expected values (this is an empty dict if compute_pickle=True)
        with open(PPO_EXPECTED_DATA_PATH, 'rb') as f:
            self.expected = pickle.load(f)

    def tearDown(self):
        # Write results of this test to disk for future reproducibility tests
        # Note: This causes unit tests to have a side effect (generally frowned upon) and only works because 
        #   unittest is single threaded. If tests were run concurrently this could result in a race condition!
        if self.compute_pickle:
            with open(PPO_EXPECTED_DATA_PATH, 'wb') as f:
                pickle.dump(self.expected, f)
        
        # Cleanup 
        if os.path.exists(self.temp_results_dir):
            shutil.rmtree(self.temp_results_dir)
        if os.path.exists(self.temp_model_dir):
            shutil.rmtree(self.temp_model_dir)
        if os.path.exists(self.temp_agent_dir):
            shutil.rmtree(self.temp_agent_dir)
        ray.shutdown()

    def test_save_load(self):
        # Train a quick self play agent for 2 iterations
        ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "results_dir": self.temp_results_dir,
                "experiment_name" : "save_load_test",
                "layout_name" : "cramped_room",
                "num_workers": 1,
                "train_batch_size": 800,
                "sgd_minibatch_size": 800,
                "num_training_iters": 2,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_potential_shaping": False,
                "evaluation_display": False,
                "verbose" : False
            },
            options={'--loglevel': 'ERROR'}
        )

        # Kill all ray processes to ensure loading works in a vaccuum
        ray.shutdown()

        # Where the agent is stored (this is kind of hardcoded, would like for it to be more easily obtainable)
        load_path = os.path.join(glob.glob(os.path.join(self.temp_results_dir, "save_load_test*"))[0], 'checkpoint_000002', 'checkpoint-2')

        # Load a dummy state
        mdp = OvercookedGridworld.from_layout_name("cramped_room")
        state = mdp.get_standard_start_state()

        # Ensure simple single-agent loading works
        agent_0 = PPOAgent.from_trainer_path(load_path)
        agent_0.reset()

        agent_1 = PPOAgent.from_trainer_path(load_path, agent_kwargs={"agent_index" : 1})
        agent_1.reset()

        # Ensure forward pass of policy network still works
        _, _ = agent_0.action(state)
        _, _ = agent_1.action(state)

        # Now let's load an SP agent pair and evaluate it
        agent = PPOAgent.from_trainer_path(load_path)
        agent_pair = AgentPair(agent, agent, allow_duplicate_agents=True)
        ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name" : "cramped_room"}, env_params={"horizon" : 400})

        # We assume no runtime errors => success, no performance consistency check for now
        ae.evaluate_agent_pair(agent_pair, 1, info=False)


    def test_ppo_sp_no_phi(self):
        # Train a self play agent for 20 iterations
        results = ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "use_potential_shaping" : False,
                "use_reward_shaping" : True,
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_potential_shaping": False,
                "evaluation_display": False,
                "verbose" : False
            },
            options={'--loglevel': 'ERROR'}
        ).result

        # Sanity check (make sure it begins to learn to receive dense reward)
        if self.check_performance:
            self.assertGreaterEqual(results['average_total_reward'], self.min_performance)

        if self.compute_pickle:
            self.expected['test_ppo_sp_no_phi'] = results
        
        # Reproducibility test
        if self.strict:
            self.assertDictEqual(results, self.expected['test_ppo_sp_no_phi'])

    def test_ppo_sp_yes_phi(self):
        # Train a self play agent for 20 iterations
        results = ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "use_potential_shaping" : True,
                "use_reward_shaping" : False,
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "use_potential_shaping": True,
                "evaluation_display": False,
                "verbose" : False
            },
            options={'--loglevel': 'ERROR'}
        ).result

        # Sanity check (make sure it begins to learn to receive dense reward)
        if self.check_performance:
            self.assertGreaterEqual(results['average_total_reward'], self.min_performance)

        if self.compute_pickle:
            self.expected['test_ppo_sp_yes_phi'] = results

        # Reproducibility test
        if self.strict:
            self.assertDictEqual(results, self.expected['test_ppo_sp_yes_phi'])


    def test_ppo_fp_sp_no_phi(self):
        # Train a self play agent for 20 iterations
        results = ex_fp.run(
            config_updates={
                "results_dir": self.temp_results_dir,
                "num_workers": 1,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "use_potential_shaping": False,
                "entropy_coeff_start": 0.0002,
                "entropy_coeff_end": 0.00005,
                "lr": 7e-4,
                "seeds": [0],
                "outer_shape": (5, 4),
                "evaluation_display": False,
                "verbose" : False
            },
            options={'--loglevel': 'ERROR'}
        ).result

        # Sanity check (make sure it begins to learn to receive dense reward)
        if self.check_performance:
            self.assertGreaterEqual(results['average_total_reward'], self.min_performance)

        if self.compute_pickle:
            self.expected['test_ppo_fp_sp_no_phi'] = results

        # Reproducibility test
        if self.strict:
            self.assertDictEqual(results, self.expected['test_ppo_fp_sp_no_phi'])


    def test_ppo_fp_sp_yes_phi(self):
        # Train a self play agent for 20 iterations
        results = ex_fp.run(
            config_updates={
                "results_dir": self.temp_results_dir,
                "num_workers": 1,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 30,
                "evaluation_interval": 10,
                "use_potential_shaping": True,
                "entropy_coeff_start": 0.0002,
                "entropy_coeff_end": 0.00005,
                "lr": 7e-4,
                "seeds": [0],
                "outer_shape": (5, 4),
                "evaluation_display": False,
                "verbose" : False
            },
            options={'--loglevel': 'ERROR'}
        ).result

        # Sanity check (make sure it begins to learn to receive dense reward)
        if self.check_performance:
            self.assertGreaterEqual(results['average_total_reward'], self.min_performance)

        if self.compute_pickle:
            self.expected['test_ppo_fp_sp_yes_phi'] = results

        # Reproducibility test
        if self.strict:
            self.assertDictEqual(results, self.expected['test_ppo_fp_sp_yes_phi'])


    def test_ppo_bc(self):
        # Train bc model
        model_dir = self.temp_model_dir
        params_to_override = { 
            "layouts" : ['inverse_marshmallow_experiment'],
            "data_path" : None,
            "epochs" : 10
        }
        bc_params = get_bc_params(**params_to_override)
        train_bc_model(model_dir, bc_params)
    
        # Train rllib model
        config_updates = { 
            "layout_name" : "inverse_marshmallow_experiment",
            "results_dir" : self.temp_results_dir, 
            "bc_schedule" : [(0.0, 0.0), (8e3, 1.0)], 
            "num_training_iters" : 20, 
            "bc_model_dir" : model_dir, 
            "evaluation_interval" : 5,
            "verbose" : False
        }
        results = ex.run(config_updates=config_updates, options={'--loglevel': 'ERROR'}).result
    
        # Sanity check
        if self.check_performance:
            self.assertGreaterEqual(results['average_total_reward'], self.min_performance)
    
        if self.compute_pickle:
            self.expected['test_ppo_bc'] = results
    
        # Reproducibility test
        if self.strict:
            self.assertDictEqual(results, self.expected['test_ppo_bc'])

    def test_ppo_bc_opt_bernoulli(self):
        # Train bc model
        model_dir = self.temp_model_dir
        params_to_override = { 
            "layouts" : ['soup_coordination'],
            "data_path" : None,
            "epochs" : 10
        }
        bc_params = get_bc_params(**params_to_override)
        train_bc_model(model_dir, bc_params)

        # Train sp "opt" model
        config_updates={
            "layout_name" : "soup_coordination",
            "experiment_name" : "my_sp_opt",
            "results_dir": self.temp_results_dir,
            "num_workers": 2,
            "num_training_iters": 10,
            "evaluation_interval": 10,
            "evaluation_display": False,
            "verbose" : False
        }
        results = ex.run(config_updates=config_updates, options={'--loglevel': 'ERROR'}).result
        opt_path = results['save_paths'][0]

        # Train rllib model
        config_updates = {
            "layout_name" : "soup_coordination",
            "bc_opt" : True,
            "opt_path" : opt_path,
            "bc_opt_cls_key" : 'bernoulli',
            "results_dir" : self.temp_results_dir, 
            "bc_schedule" : [(0.0, 0.0), (8e3, 1.0)], 
            "num_training_iters" : 20, 
            "bc_model_dir" : model_dir,
            "evaluation_interval" : 5,
            "verbose" : False
        }
        results = ex.run(config_updates=config_updates, options={'--loglevel': 'ERROR'}).result

    def test_ppo_bc_opt_counters(self):
        # Train bc model
        model_dir = self.temp_model_dir
        params_to_override = { 
            "layouts" : ['soup_coordination'],
            "data_path" : None,
            "epochs" : 10
        }
        bc_params = get_bc_params(**params_to_override)
        train_bc_model(model_dir, bc_params)

        # Train sp "opt" model
        config_updates={
            "layout_name" : "soup_coordination",
            "experiment_name" : "my_sp_opt",
            "results_dir": self.temp_results_dir,
            "num_workers": 2,
            "num_training_iters": 10,
            "evaluation_interval": 10,
            "evaluation_display": False,
            "verbose" : True
        }
        results = ex.run(config_updates=config_updates, options={'--loglevel': 'ERROR'}).result
        opt_path = results['save_paths'][0]

        # Train rllib model
        config_updates = {
            "bc_opt" : True,
            "opt_path" : opt_path,
            "bc_opt_cls_key" : "counters",
            "results_dir" : self.temp_results_dir, 
            "bc_schedule" : [(0.0, 0.0), (8e3, 1.0)], 
            "num_training_iters" : 20, 
            "bc_model_dir" : model_dir, 
            "evaluation_interval" : 5,
            "verbose" : True
        }
        results = ex.run(config_updates=config_updates, options={'--loglevel': 'ERROR'}).result

    def test_ppo_bc_opt_different_arch(self):
        # Train bc model
        model_dir = self.temp_model_dir
        params_to_override = { 
            "layouts" : ['soup_coordination'],
            "data_path" : None,
            "epochs" : 10
        }
        bc_params = get_bc_params(**params_to_override)
        train_bc_model(model_dir, bc_params)

        # Train sp "opt" model
        config_updates={
            "layout_name" : "soup_coordination",
            "experiment_name" : "my_sp_opt",
            "results_dir": self.temp_results_dir,
            "num_workers": 2,
            "num_training_iters": 10,
            "evaluation_interval": 10,
            "evaluation_display": False,
            "verbose" : False
        }
        results = ex.run(config_updates=config_updates, options={'--loglevel': 'ERROR'}).result
        opt_path = results['save_paths'][0]

        # Train rllib model (critically w/ different network architecture to make sure it doesn't break TF graphs)
        config_updates = {
            "NUM_HIDDEN_LAYERS" : 2,
            "SIZE_HIDDEN_LAYERS" : 32,
            "bc_opt" : True,
            "opt_path" : opt_path,
            "results_dir" : self.temp_results_dir, 
            "bc_schedule" : [(0.0, 0.0), (8e3, 1.0)], 
            "num_training_iters" : 20, 
            "bc_model_dir" : model_dir, 
            "evaluation_interval" : 5,
            "verbose" : False
        }
        results = ex.run(config_updates=config_updates, options={'--loglevel': 'ERROR'}).result

    def test_ficticious_self_play(self):
        # Train a ficticious-self-play agent for 20 iterations
        results = ex.run(
            config_updates={
                # Please feel free to modify the parameters below
                "ficticious_self_play" : True,
                "training_iters_per_ensemble_checkpoint" : 7,
                "training_iters_per_ensemble_sample" : 1,
                "training_iters_per_ensemble_checkpoint" : 7,
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "train_batch_size": 1600,
                "sgd_minibatch_size": 800,
                "num_training_iters": 20,
                "evaluation_interval": 10,
                "entropy_coeff_start": 0.0,
                "entropy_coeff_end": 0.0,
                "evaluation_display": False,
                "return_trainer" : True,
                "verbose" : False
            },
            options={'--loglevel': 'ERROR'}
        ).result

        trainer = results['trainers'][0]
        ensemble_ppo_policy = trainer.get_policy('ensemble_ppo')
        base_policies = ensemble_ppo_policy.base_policies

        # Ensure correct number of checkpoints added
        self.assertEqual(len(base_policies), 4)

        # Ensure we're not adding the same pointer to the same object
        self.assertIsNot(base_policies[1], base_policies[2])
        self.assertIsNot(base_policies[2], base_policies[3])

        # Ensure we aren't overriding the weights in our TF graph after storing checkpoints
        dummy_obs = trainer.env_creator(trainer.config['env_config']).reset()
        ensemble_ppo_key = [key for key in dummy_obs.keys() if key.startswith('ensemble_ppo')][0]
        dummy_obs = dummy_obs[ensemble_ppo_key]
        logits_1 = base_policies[1].compute_actions([dummy_obs])[2]['action_dist_inputs']
        logits_2 = base_policies[2].compute_actions([dummy_obs])[2]['action_dist_inputs']
        logits_3 = base_policies[3].compute_actions([dummy_obs])[2]['action_dist_inputs']
        
        self.assertFalse(np.allclose(logits_1, logits_2, atol=0.001))
        self.assertFalse(np.allclose(logits_2, logits_3, atol=0.001))
        self.assertFalse(np.allclose(logits_1, logits_3, atol=0.001))

    def test_ppo_sp_agent_serialization(self):
        # Train a self play agent for 2 iterations
        results = ex.run(
            config_updates={
                "results_dir": self.temp_results_dir,
                "num_workers": 2,
                "num_training_iters": 2,
                "evaluation_interval": 10,
                "evaluation_display": False,
                "verbose" : False
            },
            options={'--loglevel': 'ERROR'}
        ).result

        # Load agent from trainer and save in new location
        agent_save_path = os.path.join(self.temp_agent_dir, 'my_agent')
        trainer_save_path = results['save_paths'][0]
        agent = PPOAgent.from_trainer_path(trainer_save_path, 'ppo')
        agent.save(agent_save_path)

        # Delete all previous PPO data
        shutil.rmtree(self.temp_results_dir)

        # Ensure agent loading still works
        agent.load(agent_save_path)
        
    def test_ppo_bc_agent_serialization(self):
        # Train bc model
        model_dir = self.temp_model_dir
        params_to_override = { 
            "layouts" : ['soup_coordination'],
            "data_path" : None,
            "epochs" : 2
        }
        bc_params = get_bc_params(**params_to_override)
        train_bc_model(model_dir, bc_params, verbose=False)
    
        # Train rllib model
        config_updates = { 
            "layout_name" : "soup_coordination",
            "results_dir" : self.temp_results_dir, 
            "bc_schedule" : [(0.0, 0.0), (8e3, 1.0)], 
            "num_training_iters" : 20, 
            "bc_model_dir" : model_dir, 
            "evaluation_interval" : 5,
            "verbose" : False
        }
        results = ex.run(config_updates=config_updates, options={'--loglevel': 'ERROR'}).result

        # Load agent from trainer and save in new location
        agent_save_path = os.path.join(self.temp_agent_dir, 'my_agent')
        trainer_save_path = results['save_paths'][0]
        agent = PPOAgent.from_trainer_path(trainer_save_path, 'ppo_bc')
        agent.save(agent_save_path)

        # Delete all previous PPO + BC data
        shutil.rmtree(model_dir)
        shutil.rmtree(self.temp_results_dir)

        # Ensure agent loading still works
        agent.load(agent_save_path)
        
    def test_ppo_bc_opt_agent_serialization(self):
        # Train bc model
        model_dir = self.temp_model_dir
        params_to_override = { 
            "layouts" : ['asymmetric_advantages_tomato'],
            "data_path" : None,
            "epochs" : 2
        }
        bc_params = get_bc_params(**params_to_override)
        train_bc_model(model_dir, bc_params, verbose=False)

        # Train sp "opt" model
        config_updates={
            "layout_name" : "asymmetric_advantages_tomato",
            "experiment_name" : "my_sp_opt",
            "results_dir": self.temp_results_dir,
            "num_workers": 2,
            "num_training_iters": 2,
            "evaluation_interval": 10,
            "evaluation_display": False,
            "verbose" : False
        }
        results = ex.run(config_updates=config_updates, options={'--loglevel': 'ERROR'}).result
        opt_path = results['save_paths'][0]

        # Train rllib model
        config_updates = {
            "layout_name" : "asymmetric_advantages_tomato",
            "bc_opt" : True,
            "opt_path" : opt_path,
            "bc_opt_cls_key" : "counters",
            "results_dir" : self.temp_results_dir, 
            "bc_schedule" : [(0.0, 0.0), (8e3, 1.0)], 
            "num_training_iters" : 2, 
            "bc_model_dir" : model_dir, 
            "evaluation_interval" : 5,
            "verbose" : False
        }
        results = ex.run(config_updates=config_updates, options={'--loglevel': 'ERROR'}).result

        # Load agent from trainer and save in new location
        agent_save_path = os.path.join(self.temp_agent_dir, 'my_agent')
        trainer_save_path = results['save_paths'][0]
        agent = PPOAgent.from_trainer_path(trainer_save_path, 'ppo_bc_opt')
        agent.save(agent_save_path)

        # Delete all previous PPO + BC + OPT data
        shutil.rmtree(model_dir)
        shutil.rmtree(self.temp_results_dir)

        # Ensure agent loading still works
        agent.load(agent_save_path)

def _clear_pickle():
    # Write an empty dictionary to our static "expected" results location
    with open(PPO_EXPECTED_DATA_PATH, 'wb') as f:
        pickle.dump({}, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute-pickle', '-cp', action="store_true")
    parser.add_argument('--strict', '-s', action="store_true")
    parser.add_argument('--min_performance', '-mp', default=5)
    parser.add_argument('--check_performance', '-p', action="store_true")

    args = vars(parser.parse_args())

    assert not (args['compute_pickle'] and args['strict']), "Cannot compute pickle and run strict reproducibility tests at same time"
    if args['compute_pickle']:
        _clear_pickle()

    suite = unittest.TestSuite()

    # PPO SP
    suite.addTest(TestPPORllib('test_save_load', **args))
    suite.addTest(TestPPORllib('test_ppo_sp_no_phi', **args))
    suite.addTest(TestPPORllib('test_ppo_sp_yes_phi', **args))
    # suite.addTest(TestPPORllib('test_ppo_fp_sp_no_phi', **args)) # Deprecated for now
    # suite.addTest(TestPPORllib('test_ppo_fp_sp_yes_phi', **args)) # Deprecated for now

    # PPO BC
    suite.addTest(TestPPORllib('test_ppo_bc', **args))

    # PPO BC OPT
    suite.addTest(TestPPORllib('test_ppo_bc_opt_bernoulli', **args))
    suite.addTest(TestPPORllib('test_ppo_bc_opt_counters', **args))
    suite.addTest(TestPPORllib('test_ppo_bc_opt_different_arch', **args))

    # Ficticious Self-play
    suite.addTest(TestPPORllib('test_ficticious_self_play', **args))

    # Agent stuff
    suite.addTest(TestPPORllib('test_ppo_sp_agent_serialization', **args))
    suite.addTest(TestPPORllib('test_ppo_bc_agent_serialization', **args))
    suite.addTest(TestPPORllib('test_ppo_bc_opt_agent_serialization', **args))

    success = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    sys.exit(not success)
        


