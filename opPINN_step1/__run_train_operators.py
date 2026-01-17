"""
# batch_size=300
for SEED in 0
do
    echo "Starting Jobs for SEED: $SEED"
    
    for gamma in -3.0 -2.0 -1.0 0.0 1.0
    do
        # Train the operators D and F
        nohup python run_trainer.py \
            --seed $SEED \
            --cuda_index 2 \
            --dimension 2 \
            --sample_v 64 \
            --gamma $gamma \
            --op D \
            > output__2d__gamma${gamma%1f}__op_D__seed${SEED}.log &
        nohup python run_trainer.py \
            --seed $SEED \
            --cuda_index 3 \
            --dimension 2 \
            --sample_v 64 \
            --gamma $gamma \
            --op F \
            > output__2d__gamma${gamma%1f}__op_F__seed${SEED}.log &
        wait
    done
done

echo "All jobs submitted!"
"""
import  argparse
import  subprocess
from    math        import  pi

import  sys
sys.path.append('..')
from    config.op_config    import  *


LIST_SEEDS:         list[int]   = [0]
LIST_GAMMA_VMAX:    list[tuple[float, float]] = [
    (-3.0, 5.0),
    (-2.0, 5.0),
    (-1.0, 5.0),
    (0.0, 5.0),
    (0.0, 2*pi)
]


parser = argparse.ArgumentParser()
parser.add_argument('--cuda_index', type=int, default=3, help='CUDA device index.')
parser.add_argument('--dimension', type=int, help='Dimension of the problem.')
parser.add_argument('--sample_v', type=int, default=64, help='The sampling resolution in the velocity space.')
parser.add_argument('--op', type=str, choices=['D', 'F'], help='The operator to be trained.')
parser.add_argument('--batch_size', type=int, default=OP__BATCH_SIZE, help='Batch size for training.')
args = parser.parse_args()


subprocess.run(
    [
        "python", "train_operators.py",
        
        "--seed",           str(args.seed),
        "--cuda_index",     str(args.cuda_index),
        
        "--dimension",      str(args.dimension),
        "--resolution",     str(args.sample_v),
        "--gamma",          str(args.gamma),
        
        "--operator",       str(args.op),
        
        "--num_epochs",     str(OP__NUM_EPOCHS),
        "--learning_rate",  str(OP__LEARNING_RATE),
        "--batch_size",     str(OP__BATCH_SIZE),
        
        "--period_report",  str(OP__PERIOD_REPORT),
        "--period_backup",  str(OP__PERIOD_BACKUP),
    ]
)