from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import Agent, AgentPair
from human_aware_rl.rllib.utils import softmax, get_base_ae
from human_aware_rl.rllib.rllib import RlLibAgent
from human_aware_rl.rllib.rllib import load_agent as ppo_load_agent
from human_aware_rl.rllib.pbt_rllib import load_agent as pbt_load_agent
from human_aware_rl.rllib.fcp_rllib import load_agent as fcp_load_agent
import numpy as np
import os, dill
import ray


def load_agent(agent_info, agent_index):
    c_dir = agent_info['checkpoint_dir']
    c_num = agent_info['checkpoint_num']
    agent_path = os.path.join("/home/junyong/ray_results/", c_dir, f"checkpoint_{c_num}", f"checkpoint-{c_num}")

    agent_type = agent_info['agent_type']
    policy_id = agent_info['policy_id']
    if agent_type == "ppo":
        agent = ppo_load_agent(agent_path, policy_id, agent_index)
    elif agent_type == "pbt":
        agent = pbt_load_agent(agent_path, policy_id, agent_index)
    elif agent_type == "fcp":
        agent = fcp_load_agent(agent_path, policy_id, agent_index)
    
    return agent

def run_evaluation(mdp_params, eval_params, main_agent_info, partner_agents_info):
    final_results = []

    evaluator = get_base_ae(mdp_params, {"horizon" : eval_params['ep_length'], "num_mdp":1}, None)

    main_agent = load_agent(main_agent_info, 0)

    for partner_name, partner_agent_info in partner_agents_info.items():
        partner_agent = load_agent(partner_agent_info, 1)

        # Compute rollouts
        results = evaluator.evaluate_agent_pair(AgentPair(main_agent, partner_agent),
                                                num_games=eval_params['num_games'],
                                                display=eval_params['display'],
                                                dir=eval_params['store_dir'],
                                                display_phi=eval_params['display_phi'],
                                                info=False)

        final_results.append((partner_name, np.mean(results['ep_returns'])))

    print("==========================================")
    score_sum = 0
    for name, score in final_results:
        print(f"{name}: {score}")
        score_sum += score
    print(f"mean: {score_sum/len(final_results)}")
    print("==========================================")

def main():
    mdp_params = {
        "layout_name": "modified_room",
        "shaped_reward_params": {
            "add_ingredient_to_container": 3,
            "move_food_from_container": 5,
            "throw_away_food": 2,
            "throw_away_container": 3,
            "place_object": 2,
            "pickup_dispenser": 1,
            "pickup_object": 2,
            "start_cooking": 6
        }
    }
    
    evaluation_params = {
        "ep_length" : 1000,
        "num_games" : 10,
        "display" : False,
        "store_dir" : None,
        "display_phi" : False
    }

    main_agent_info = {
        'checkpoint_dir' : 'fcp_first',
        'checkpoint_num' : 10000,
        'agent_type' : 'fcp',
        'policy_id' : 'fcp'
    }
    # main_agent_info = {
    #     'checkpoint_dir' : 'fcp_second',
    #     'checkpoint_num' : 20000,
    #     'agent_type' : 'fcp',
    #     'policy_id' : 'fcp'
    # }
    # main_agent_info = {
    #     'checkpoint_dir' : 'sp_fcp_2228',
    #     'checkpoint_num' : 10000,
    #     'agent_type' : 'ppo',
    #     'policy_id' : 'ppo'
    # }

    partner_agents_info = {
        'random 1' : {
            'checkpoint_dir' : 'test/sp_2228_2022-05-16_11-00-58ony63alf',
            'checkpoint_num' : 1,
            'agent_type' : 'ppo',
            'policy_id' : 'ppo'
        },
        'random 2' : {
            'checkpoint_dir' : 'test/sp_2228_2022-05-16_11-02-5381dddn1w',
            'checkpoint_num' : 1,
            'agent_type' : 'ppo',
            'policy_id' : 'ppo'
        },
        'beginner 1' : {
            'checkpoint_dir' : 'test/sp_2228_2022-05-16_11-00-58ony63alf',
            'checkpoint_num' : 401,
            'agent_type' : 'ppo',
            'policy_id' : 'ppo'
        },
        'beginner 2' : {
            'checkpoint_dir' : 'test/sp_2228_2022-05-16_11-02-5381dddn1w',
            'checkpoint_num' : 501,
            'agent_type' : 'ppo',
            'policy_id' : 'ppo'
        },
        'mediocre 1' : {
            'checkpoint_dir' : 'test/sp_2228_2022-05-16_11-00-58ony63alf',
            'checkpoint_num' : 501,
            'agent_type' : 'ppo',
            'policy_id' : 'ppo'
        },
        'mediocre 2' : {
            'checkpoint_dir' : 'test/sp_2228_2022-05-16_11-02-5381dddn1w',
            'checkpoint_num' : 601,
            'agent_type' : 'ppo',
            'policy_id' : 'ppo'
        },
        'skillful 1' : {
            'checkpoint_dir' : 'test/sp_2228_2022-05-16_11-00-58ony63alf',
            'checkpoint_num' : 10000,
            'agent_type' : 'ppo',
            'policy_id' : 'ppo'
        },
        'skillful 2' : {
            'checkpoint_dir' : 'test/sp_2228_2022-05-16_11-02-5381dddn1w',
            'checkpoint_num' : 10000,
            'agent_type' : 'ppo',
            'policy_id' : 'ppo'
        }
    }

    run_evaluation(mdp_params, evaluation_params, main_agent_info, partner_agents_info)

if __name__ == '__main__':
    main()