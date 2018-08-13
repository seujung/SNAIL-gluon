import mxnet as mx
import argparse
from trainer import Train

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epoches', type=int, default=200)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--input_dims', type=int, default=64)
    parser.add_argument('--download', type=bool, default=False)
    parser.add_argument('--GPU_COUNT', type=int, default=2)
    config = parser.parse_args()
    
    trainer = Train(config)
    
    trainer.train()
    if (config.generation):
        trainer.generation()

if __name__ =="__main__":
    main()