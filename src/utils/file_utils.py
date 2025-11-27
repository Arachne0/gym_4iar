import os
import wandb


def initialize_wandb(args):
    common_config = {
        "entity": "hails",
        "project": "gym_4iar",
        "config": args.__dict__
    }
    if args.rl_model in ["AC", "EQRAC"]:
        run_name = f"{args.rl_model}-nmcts{args.n_playout}"
    elif args.rl_model == "QRAC":
        run_name = f"{args.rl_model}-nmcts{args.n_playout}-quantiles{args.quantiles}"
    elif args.rl_model in ["DQN", "EQRDQN"]:
        run_name = f"{args.rl_model}-nmcts{args.n_playout}-eps{args.epsilon}"
    elif args.rl_model == "QRDQN":
        run_name = f"{args.rl_model}-nmcts{args.n_playout}-quantiles{args.quantiles}-eps{args.epsilon}"
    else:
        raise ValueError("Model is not defined")

    wandb.init(name=run_name, **common_config)


def create_models(args, i=None):
    """
    Generate training and evaluation model file paths dynamically based on the rl_model and parameters.
    """
    # Define model-specific path structures
    model_params = {
        "DQN": f"_nmcts{args.n_playout}_eps{args.epsilon}",
        "QRDQN": f"_nmcts{args.n_playout}_quantiles{args.quantiles}_eps{args.epsilon}",
        "EQRDQN": f"_nmcts{args.n_playout}_eps{args.epsilon}",
        "AC": f"_nmcts{args.n_playout}",
        "QRAC": f"_nmcts{args.n_playout}_quantiles{args.quantiles}",
        "EQRAC": f"_nmcts{args.n_playout}"
    }
    if args.rl_model not in model_params:
        raise ValueError("Model is not defined")

    # Construct the specific path part for the model
    specific_path = model_params[args.rl_model]
    filename = f"{i + 1:03d}.pth"

    # Generate full paths
    model_file = f"models82/Training/{args.rl_model}{specific_path}/{filename}"
    eval_model_file = f"models82/Eval/{args.rl_model}{specific_path}/{filename}"

    return model_file, eval_model_file


def get_existing_files(args):
    """
    Retrieve a list of existing file indices based on the model type and parameters.
    """
    base_path = "models82/Training"
    if args.rl_model == "DQN":
        path = f"{base_path}/{args.rl_model}_nmcts{args.n_playout}_eps{args.epsilon}"
    elif args.rl_model == "QRDQN":
        path = f"{base_path}/{args.rl_model}_nmcts{args.n_playout}_quantiles{args.quantiles}_eps{args.epsilon}"
    elif args.rl_model == "EQRDQN":
        path = f"{base_path}/{args.rl_model}_nmcts{args.n_playout}_eps{args.epsilon}"
    elif args.rl_model == "AC":
        path = f"{base_path}/{args.rl_model}_nmcts{args.n_playout}"
    elif args.rl_model == "QRAC":
        path = f"{base_path}/{args.rl_model}_nmcts{args.n_playout}_quantiles{args.quantiles}"
    elif args.rl_model == "EQRAC":
        path = f"{base_path}/{args.rl_model}_nmcts{args.n_playout}"
    else:
        raise ValueError("Model is not defined")

    # Fetch files and extract indices
    return [
        int(file.split('.')[0])     
        for file in os.listdir(path)
        if file.endswith('.pth')    
    ]