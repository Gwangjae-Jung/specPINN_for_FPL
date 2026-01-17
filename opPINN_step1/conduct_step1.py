import  argparse
import  subprocess
from    math           import   pi
from    typing         import   TypeAlias, Literal


Type_gamma_coeff_vmax: TypeAlias = tuple[float, float, float]
InitialConditions: TypeAlias = Literal["bimaxwellian", "maxwellian", "bkw"]
cfg1: Type_gamma_coeff_vmax = (-3.0, 5.0, 5.0)
cfg2: Type_gamma_coeff_vmax = (-2.0, 3.0, 5.0)
cfg3: Type_gamma_coeff_vmax = (-1.0, 1.0, 5.0)
cfg4: Type_gamma_coeff_vmax = (0.0, 5/16, 5.0)
cfg5: Type_gamma_coeff_vmax = (0.0, 5/16, 2*pi)
gamma_coeff_vmax: list[Type_gamma_coeff_vmax] = [cfg1, cfg2, cfg3, cfg4, cfg5]
cfg_init: dict[Type_gamma_coeff_vmax, list[InitialConditions]] = {
    cfg1: ["bimaxwellian"],
    cfg2: ["bimaxwellian"],
    cfg3: ["bimaxwellian"],
    cfg4: ["bimaxwellian"],
    cfg5: ["maxwellian", "bkw"],
}


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
parser.add_argument('--cuda_index_1', type=int, default=2, help="The index of the first CUDA device, which will be used to generate datasets and train the operator 'D'.")
parser.add_argument('--cuda_index_2', type=int, default=3, help="The index of the first CUDA device, which will be used to train the operator 'F'.")
parser.add_argument('--dimension', type=int, default=2, help='Dimension of the velocity space.')
parser.add_argument('--resolution', type=int, default=64, help='Number of grid points per dimension.')
parser.add_argument('--density', type=float, default=0.2, help='The density of the input distribution functions.')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training.')
parser.add_argument('--num_epochs', type=int, default=int(5e4), help='Number of training epochs.')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer.')
parser.add_argument('--scheduler_step', type=int, default=2000, help='Step size for the learning rate scheduler.')
parser.add_argument('--scheduler_decay', type=float, default=0.9, help='Decay factor for the learning rate scheduler.')
parser.add_argument('--period_report', type=int, default=100, help='Period of reporting during training.')
parser.add_argument('--period_backup', type=int, default=1000, help='Period of backing up the model during training.')

parser.add_argument('--generate_only', action='store_true', help='If set, only generate datasets without training operators.')
parser.add_argument('--train_only', action='store_true', help='If set, only train operators without generating datasets.')
args = parser.parse_args()
if args.generate_only and args.train_only:
    raise ValueError("Cannot set both --generate_only and --train_only.")
generate_only:  bool    = args.generate_only
train_only:     bool    = args.train_only
both_steps:     bool    = (not generate_only) and (not train_only)

# Part 1. Generate datasets
if both_steps or generate_only:
    print("Generating datasets...")
    for cfg in gamma_coeff_vmax:
        gamma, coeff, v_max = cfg
        subprocess.run([
            "python3", "_generate_dataset.py",
            "--cuda_index", str(args.cuda_index_1),
            "--dimension", str(args.dimension),
            "--resolution", str(args.resolution),
            "--v_max", str(v_max),
            "--gamma", str(gamma),
        ])
    print("Datasets generation completed.\n")


# Part 2. Train operators D and F
if both_steps or train_only:
    print("Training operators D and F...")
    def get_trainer_args(gamma: float, v_max: float, operator: Literal['D', 'F']) -> list[str]:
        base = [
            "python3", "_train_operators.py",
            "--seed", str(args.seed),
            # "--cuda_index", str(args.cuda_index_1),
            "--dimension", str(args.dimension),
            "--resolution", str(args.resolution),
            "--gamma", str(gamma),
            "--v_max", str(v_max),
            "--density", str(args.density),
            # "--operator", "D",
            "--batch_size", str(args.batch_size),
            "--num_epochs", str(args.num_epochs),
            "--learning_rate", str(args.learning_rate),
            "--scheduler_step", str(args.scheduler_step),
            "--scheduler_decay", str(args.scheduler_decay),
            "--period_report", str(args.period_report),
            "--period_backup", str(args.period_backup),
        ]
        if operator=='D':
            base += ["--operator", "D"]
            base += ["--cuda_index", str(args.cuda_index_1)]
        elif operator=='F':
            base += ["--operator", "F"]
            base += ["--cuda_index", str(args.cuda_index_2)]
        else:
            raise ValueError(f"Unknown operator: {operator}")
        return base
        

    for cfg in gamma_coeff_vmax:
        gamma, coeff, v_max = cfg
        for init_cond in cfg_init[cfg]:
            print(f"Training operators for gamma={gamma}, v_max={v_max}, initial condition={init_cond}")
            subprocess.run(get_trainer_args(gamma, v_max, 'D'))
            subprocess.run(get_trainer_args(gamma, v_max, 'F'))
            print("Finished.\n")