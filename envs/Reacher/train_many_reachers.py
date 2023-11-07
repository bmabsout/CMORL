import argparse
from . import train_reacher
from cmorl.utils import args_utils
from pathlib import Path


def train_shrinking_circle():
    for i in range(6):
        train_reacher.train_reacher(train_reacher.parse_arguments(["reacher", "--distance", f"{0.2*((i+1.0)/6.0)}", "-r"]))

def fine_tune():
    for i in range(6):
        cmd_args = train_reacher.parse_args([
            "reacher",
            "--epochs", "10",
            "--distance", f"{0.2*((i+1.0)/6.0)}",
            "--prev_folder", 'reacher/d:0.2,e:50,l:0.003,x:PKUZRZSQ7CS6S7O/seeds/601909/epochs/49'
        ])
        print(cmd_args)
        train_reacher.train(cmd_args)

def many_fine_tunes(anchored = True):
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', nargs="+", type=str, help='location of training runs to fine tune')
    serializer = train_reacher.reacher_serializer()
    serializer.abbrev_to_args['e'] = args_utils.Serialized_Argument(name='--epochs', type=int, default=10)
    serializer.abbrev_to_args['d'] = args_utils.Serialized_Argument(name='--distance', type=int, default=0.1)
    serializer.add_serialized_args_to_parser(parser)
    cmd_args = parser.parse_args()
    for folder in cmd_args.folders:
        cmd_args.prev_folder = Path(folder)
        train_reacher.train(cmd_args, serializer)



def train_many_full_circles():
    for i in range(6):
        serializer = train_reacher.reacher_serializer()
        cmd_args = args_utils.parse_arguments(serializer, args=["-r"])
        train_reacher.train(cmd_args, serializer)


if __name__ == "__main__":
    many_fine_tunes()
    # train_many_full_circles()
