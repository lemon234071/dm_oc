import random
import torch
import logging
import argparse
import numpy as np

from data_utils.data_process import get_data
from weighting.lm import LMManager


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2020)
logging.basicConfig(level=logging.INFO, format='\n %(asctime)s - %(levelname)s - %(message)s')  # - %(name)s
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('device:', device)

# torch.backends.cudnn.enabled = True
#
# torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

# data
parser.add_argument("--dataset_path", type=str, default="data/Daily/daily_tup.json",
                    help="Path or url of the dataset. If empty download accroding to dataset.")
parser.add_argument("--dataset", type=str, default="Daily")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache',
                    help="Path or url of the dataset cache")
parser.add_argument("--save_dir", type=str, default="checkpoints")

# model
parser.add_argument("--model_checkpoint", type=str, default='pytorch_models/small_gpt2/',
                    help="Path or URL of the model")
parser.add_argument("--infer_from", type=str, default='',
                    help="Path or URL of the model")
parser.add_argument('--pretrained', action='store_true',
                    help="If False train from scratch")

# training
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--min_epochs', default=0, type=int)
parser.add_argument('--pretrain_epochs', default=0, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--val_batch_size', default=4, type=int)
parser.add_argument('--report_every', default=100, type=int)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Accumulate gradients on several steps")
parser.add_argument("--max_norm", type=float, default=1.0,
                    help="Clipping gradient norm")
parser.add_argument("--max_val_step", type=float, default=2)
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--warmup_steps", type=int, default=5000)
parser.add_argument("--lr_schedule", type=str,
                    choices=['noam', 'noamwd', 'BERT', 'None'], default='noam')
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--num_optim_steps", type=int, default=1000000,
                    help="new API specifies num update steps")

# weighting
parser.add_argument("--w_decay", default=10., type=float)
parser.add_argument("--w_init", default=0., type=float)
parser.add_argument('--norm_fn', choices=['linear', 'softmax'])

# decoding
parser.add_argument('--max_length', default=30, type=int)
parser.add_argument("--min_length", type=int, default=1,
                    help="Minimum length of the output utterances")
parser.add_argument("--top_k", type=int, default=0,
                    help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.0,
                    help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
parser.add_argument("--temperature", type=int, default=1,
                    help="Sampling softmax temperature")
parser.add_argument("--outpath", type=str, default="result.txt")

args = parser.parse_args()
print(args)


def main():
    manager = LMManager(
        args=args,
        pretrained=args.pretrained,
        model_checkpoint=args.model_checkpoint if args.infer_from is "" else args.infer_from,
        report_every=args.report_every,
        ren=False,
        norm_fn=args.norm_fn,
        device=device)
    manager.get_optimizer(learning_rate=args.learning_rate)

    data = get_data(manager.tokenizer, args.dataset_path, args.dataset_cache, args.dataset)
    if "test" not in data:
        data["test"] = data["valid"].copy()
    manager.load_data(
        'train', data['train'], args.batch_size, shuffle=True)
    manager.load_data(
        'dev_shuffle', data['valid'], args.val_batch_size, shuffle=True)
    manager.load_data(
        'dev', data['valid'], args.batch_size, shuffle=False)
    manager.load_data(
        'test', data['test'], args.batch_size, shuffle=False)

    if args.infer_from is not "":
        manager.infer(data['test'], args.outpath)
        exit()

    manager.init_weights(  # len(data['train']),
        w_init=args.w_init,
        w_decay=args.w_decay)

    print('=' * 60, '\n', 'Pre-training', '\n', '=' * 60, sep='')
    for epoch in range(args.pretrain_epochs):
        manager.pretrain_epoch()
        dev_ppl = manager.evaluate('dev')
        test_ppl = manager.evaluate('test')
        manager.save(epoch)

        print('Pre-train Epoch {}, Dev PPL: {:.4f}, Test PPL: {:.4f}'.format(
            epoch, dev_ppl, test_ppl))

    print('=' * 60, '\n', 'Training', '\n', '=' * 60, sep='')
    best_dev_ppl, final_test_ppl = 10000., 10000.
    for epoch in range(args.epochs):
        manager.train_epoch()
        dev_ppl = manager.evaluate('dev')
        manager.save(epoch + args.pretrain_epochs)

        if epoch >= args.min_epochs:
            do_test = (dev_ppl < best_dev_ppl)
            best_dev_ppl = min(best_dev_ppl, dev_ppl)
        else:
            do_test = False

        print('Epoch {}, Dev PPL: {:.4f}, Best Ever: {:.4f}'.format(
            epoch, dev_ppl, best_dev_ppl))

        if do_test:
            final_test_ppl = manager.evaluate('test')
            print('Test PPL: {:.4f}'.format(final_test_ppl))
    print('Final Dev PPL: {:.4f}, Final Test PPL: {:.4f}'.format(
        best_dev_ppl, final_test_ppl))


if __name__ == '__main__':
    main()
