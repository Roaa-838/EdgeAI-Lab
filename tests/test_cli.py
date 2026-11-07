from edgeai.cli import build_parser


def test_cli_status():
    assert build_parser().parse_args(['status']).command=='status'

def test_cli_benchmark_defaults():
    args=build_parser().parse_args(['benchmark'])
    assert args.batch_sizes==[1,8] and args.warmup==10

def test_cli_train_default_scratch():
    args=build_parser().parse_args(['train'])
    assert args.arch=='mobilenet_v3_small' and args.epochs==8
