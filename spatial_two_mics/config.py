import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# my_dregon detaset
MY_DREGON_PATH = os.path.join(BASE_PATH, 'dataset', 'MY_DREGON')
MY_DREGON_PATH2 = MY_DREGON_PATH

# here the models and the experiment's results will be saved
BASE_RESULTS = os.path.join(BASE_PATH, 'experiment_output')
MODELS_DIR = os.path.join(BASE_RESULTS,'model')
RESULTS_DIR = os.path.join(BASE_RESULTS,"results")
MODELS_RAW_PHASE_DIR = os.path.join(BASE_RESULTS, "raw_phase")
MODELS_GROUND_TRUTH = os.path.join(BASE_RESULTS, "ground_truth")
FINAL_RESULTS_DIR = os.path.join(BASE_RESULTS, "final_results")

# This is from the original work- 
TIMIT_PATH = "/mnt/data/Speech/timit-wav"
DATASETS_DIR = "/mnt/nvme/spatial_two_mics_data/"