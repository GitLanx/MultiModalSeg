import torch
import torch.nn as nn


def get_optimizer(args, model):
    if args.optim.lower() == 'sgd':
        if args.model.lower() in ['fcn32s', 'fcn8s']:
            optim = fcn_optim(model, args)
        elif args.model.lower() in ['fcn8smulti-gnn', 'fcn8smulti-gnn2']:
            optim = gnn_optim(model, args)
        elif args.model.lower() in ['multi-gnn1']:
            optim = deep_gnn_optim(model, args)
        elif args.model.lower() in ['fcn8smulti']:
            optim = fcn_multi_optim(model, args)
        else:
            optim = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.beta1,
                weight_decay=args.weight_decay)
    
    return optim

# FCN
def gnn_optim(model, args):
    optim = torch.optim.SGD(
        [{'params': model.get_parameters(double=False)},
         {'params': model.get_parameters(double=True), 'lr': args.lr * 10}],
         lr=args.lr,
         momentum=args.beta1,
         weight_decay=args.weight_decay)
    return optim

# Deeplab
def deep_gnn_optim(model, args):
    # optim = torch.optim.SGD(
    #     [{'params': model.get_parameters(score=False)},
    #      {'params': model.get_parameters(score=True), 'lr': args.lr * 1}],
    #      lr=args.lr,
    #      momentum=args.beta1,
    #      weight_decay=args.weight_decay)
    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return optim


def fcn_multi_optim(model, args):
    optim = torch.optim.SGD(
        [{'params': model.get_parameters(double=False)},
         {'params': model.get_parameters(double=True), 'lr': args.lr * 10}],
        lr=args.lr,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return optim

def fcn_optim(model, args):
    """optimizer for fcn32s and fcn8s
    """
    optim = torch.optim.SGD(
        model.get_parameters(),
        lr=args.lr,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return optim
