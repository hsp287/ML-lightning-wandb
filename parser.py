import argparse

def default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default='mnist')
    parser.add_argument('--best_metric', type=str, default='val_acc')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_file', type=str, default='mnist_data.h5')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args([])
    return args