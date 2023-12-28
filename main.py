import os
import sys
import torch
from tools import train_net, test_net, infer_net
from utils import parser
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

def main():

    args = parser.get_args()
    parser.setup(args)    
    print(args)
    if args.test:
        test_net(args) # 测试函数
    else:
        train_net(args) # 训练函数

    # infer_net(args)

if __name__ == '__main__':
    main()