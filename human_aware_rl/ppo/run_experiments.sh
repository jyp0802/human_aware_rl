#!/usr/bin/env bash
# python ppo_rllib_client.py with seeds="[2229]" lr=5e-4 reward_shaping_horizon=2.5e6 vf_loss_coeff=1e-4 num_training_iters=500 layout_name="cramped_room" experiment_name="ppo_sp_higher_S_hor_S_final_1e-2" entropy_coeff_horizon=3e6 entropy_coeff_end=1e-2
# python ppo_rllib_client.py with seeds="[2229]" lr=5e-4 reward_shaping_horizon=2.5e6 vf_loss_coeff=1e-4 num_training_iters=500 layout_name="cramped_room" experiment_name="ppo_sp_higher_S_hor_S_final_1e-3" entropy_coeff_horizon=3e6 entropy_coeff_end=1e-3
# python ppo_rllib_client.py with seeds="[2229]" lr=5e-4 reward_shaping_horizon=2.5e6 vf_loss_coeff=1e-4 num_training_iters=500 layout_name="cramped_room" experiment_name="ppo_sp_higher_S_hor_S_final_1e-4" entropy_coeff_horizon=3e6 entropy_coeff_end=1e-4
# python ppo_rllib_client.py with seeds="[2229]" lr=5e-4 reward_shaping_horizon=2.5e6 vf_loss_coeff=1e-4 num_training_iters=500 layout_name="cramped_room" experiment_name="ppo_sp_S_final_1e-2" entropy_coeff_end=1e-2
# python ppo_rllib_client.py with seeds="[2229]" lr=5e-4 reward_shaping_horizon=2.5e6 vf_loss_coeff=1e-4 num_training_iters=500 layout_name="cramped_room" experiment_name="ppo_sp_S_final_1e-3" entropy_coeff_end=1e-3
# python ppo_rllib_client.py with seeds="[2229]" lr=5e-4 reward_shaping_horizon=2.5e6 vf_loss_coeff=1e-4 num_training_iters=500 layout_name="cramped_room" experiment_name="ppo_sp_S_final_1e-4" entropy_coeff_end=1e-4
# python ppo_rllib_client.py with temp_dir=/tmp/nathan_ray seeds="[2229, 7649, 7225, 9807, 386]" lr=6e-4 reward_shaping_horizon=3.5e6 vf_loss_coeff=1e-4 num_training_iters=833 layout_name="coordination_ring" experiment_name="ppo_sp_coord_ring" 
# python ppo_rllib_client.py with temp_dir=/tmp/nathan_ray seeds="[2229, 7649, 7225, 9807, 386]" lr=8e-4 reward_shaping_horizon=2.5e6 vf_loss_coeff=1e-4 num_training_iters=667 layout_name="counter_circuit" experiment_name="ppo_sp_counter_circuit"
# python ppo_rllib_client.py -tmp /tmp/nathan_ray -s 2229 7649 7225 9807 386 -lr 8e-4 -r 2.5e6 --gpus 1 -vf 0.5 -n 625 -l "forced_coordination"   -en "ppo_sp_forced_coord"
# python ppo_rllib_client.py -tmp /tmp/nathan_ray -s 2229 7649 7225 9807 386 -lr 1e-3 -r 2.5e6 --gpus 1 -vf 0.5 -n 583 -l "asymmetric_advantages"   -en "ppo_sp_asymm_advs"

python ppo_rllib_client.py with seeds="[2229]" lr=5e-4 num_training_iters=10000 layout_name="modified_room" experiment_name="sp"