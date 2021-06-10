import os

_curr_directory = os.path.dirname(os.path.abspath(__file__))

# Root dir where all hunan data is located
HUMAN_DATA_DIR = os.path.join(_curr_directory, "human_data")

# Paths to pre-processed data
CLEAN_HUMAN_DATA_DIR = os.path.join(HUMAN_DATA_DIR, "cleaned")
CLEAN_2020_HUMAN_DATA_ALL = os.path.join(CLEAN_HUMAN_DATA_DIR, "2020_hh_trials_all.pickle")
CLEAN_2020_HUMAN_DATA_TRAIN = os.path.join(CLEAN_HUMAN_DATA_DIR, "2020_hh_trials_train.pickle")
CLEAN_2020_HUMAN_DATA_TEST = os.path.join(CLEAN_HUMAN_DATA_DIR, "2020_hh_trials_test.pickle")
CLEAN_2019_HUMAN_DATA_ALL = os.path.join(CLEAN_HUMAN_DATA_DIR, "2019_hh_trials_all.pickle")
CLEAN_2019_HUMAN_DATA_TRAIN = os.path.join(CLEAN_HUMAN_DATA_DIR, "2019_hh_trials_train.pickle")
CLEAN_2019_HUMAN_DATA_TEST = os.path.join(CLEAN_HUMAN_DATA_DIR, "2019_hh_trials_test.pickle")

# Paths to raw data
RAW_HUMAN_DATA_DIR = os.path.join(HUMAN_DATA_DIR, 'raw')
RAW_2020_HUMAN_DATA = os.path.join(RAW_HUMAN_DATA_DIR, '2020_hh_trials.csv')
RAW_2019_HUMAN_DATA = os.path.join(RAW_HUMAN_DATA_DIR, '2019_hh_trials.csv')

# Human data tests (smaller datasets for more efficient sanity checks)
DUMMY_HUMAN_DATA_DIR = os.path.join(HUMAN_DATA_DIR, "dummy")
DUMMY_2020_CLEAN_HUMAN_DATA_PATH = os.path.join(DUMMY_HUMAN_DATA_DIR, "dummy_2020_hh_trials.pickle")
DUMMY_2020_RAW_HUMAN_DATA_PATH = os.path.join(DUMMY_HUMAN_DATA_DIR, "dummy_2020_hh_trials.csv")
DUMMY_2019_CLEAN_HUMAN_DATA_PATH = os.path.join(DUMMY_HUMAN_DATA_DIR, "dummy_2019_hh_trials_all.pickle")
DUMMY_2019_RAW_HUMAN_DATA_PATH = os.path.join(DUMMY_HUMAN_DATA_DIR, "dummy_2019_hh_trials.csv")


# Expected values for reproducibility unit tests
TESTING_DATA_DIR = os.path.join(_curr_directory, "testing_data")
BC_EXPECTED_DATA_PATH = os.path.join(TESTING_DATA_DIR, "bc", "expected.pickle")
PPO_EXPECTED_DATA_PATH = os.path.join(TESTING_DATA_DIR, "ppo", "expected.pickle")
RLLIB_TRAINER_PATH = os.path.join(TESTING_DATA_DIR, 'serialization_forward_compat', 'checkpoint_000005', 'checkpoint-5')


# Human data constants
OLD_SCHEMA = set(['Unnamed: 0', 'Unnamed: 0.1', 'cur_gameloop', 'datetime', 'is_leader', 'joint_action', 'layout', 
              'layout_name', 'next_state', 'reward', 'round_num', 'round_type', 'score', 'state', 'time_elapsed', 
              'time_left', 'is_wait', 'completed', 'run', 'workerid_num'])

NEW_SCHEMA = set(['state', 'joint_action', 'reward', 'time_left', 'score', 'time_elapsed', 'cur_gameloop', 'layout', 
              'layout_name', 'trial_id', 'player_0_id', 'player_1_id', 'player_0_is_human', 'player_1_is_human'])

LAYOUTS_WITH_DATA_2019 = set(['asymmetric_advantages', 'coordination_ring', 'cramped_room', 'random0', 'random3'])
LAYOUTS_WITH_DATA_2020 = set(['asymmetric_advantages_tomato', 'counter_circuit', 'cramped_corridor', 'inverse_marshmallow_experiment', 'marshmallow_experiment', 'marshmallow_experiment_coordination', 'soup_coordination', 'you_shall_not_pass'])

LAYOUTS_WITH_DATA = LAYOUTS_WITH_DATA_2019.union(LAYOUTS_WITH_DATA_2020)