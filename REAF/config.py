import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--csv_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='REAF')
    parser.add_argument('--model_path', type=str, default='./weight')
    parser.add_argument('--load_model_path', type=str, default='')
    parser.add_argument('--writer_comment', type=str, default='BUSI')
    parser.add_argument('--save_model', type=bool, default=True)

    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--class_num', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--log_step', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--loss_function', type=str, default='CE')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--warmup_decay', type=float, default=0.01)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--step', type=int, default=5)

    parser.add_argument('--mask_path', type=str, default='')
    parser.add_argument('--train_csv_path', type=str, default='')
    parser.add_argument('--test_csv_path', type=str, default='')

    config = parser.parse_args()
    return config