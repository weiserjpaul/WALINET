

params ={}

# Path
params["model_name"] =  "WALINET" #"test"# "EXP_1" 
params["path_to_model"] = "models/" + params["model_name"] + "/"
params["path_to_data"] = "data/"

params["train_subjects"]=["vol22"] 

params["val_subjects"]=['vol22']

# Train Params
params["gpu"]=0
params["batch_size"] = 256
params["num_worker"] = 20
params["lr"] = 0.0001
params["epochs"]=10
params["verbose"] = params["model_name"] == "test" #True/False
params["n_batches"] = -1 # -1: iterate through whole dataset 
params["n_val_batches"] = -1 # -1: iterate through whole dataset 

# LR Scheduler
params["patience"] = 20

# Model Params
params["nLayers"] = 4
params["nFilters"] = 8
params["in_channels"] = 2 
params["out_channels"] = 2
params["dropout"] = 0.0

# Configurations
params["clean_model"] = False # Overwrites exisiting models. Exclusively removes models called "test"
params["train"] = False
params["predict"] = True
