import sys
import os

# setting of workspace. change based on your filesystem location.
# might not be necessary if using IDE instead of command line.
if os.name == 'posix':
    # Google Colab (Linux)
    sys.path.insert(0, '/content/KNnice/src')
else:
    # Windows
    sys.path.insert(0, 'C:\\Users\\Samuel\\Desktop\\KNnice\\src')

import sys
import torch
import prep
from lib import paths
from lib.arg_parse import parse_arguments

if __name__ == '__main__':
    args = parse_arguments()
    paths.init_dirs()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using: {device}")

    if args.train:
        print(f"\tin train")
        prep.train_net(args, device)

sys.exit(0)
