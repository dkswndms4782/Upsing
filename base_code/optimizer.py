from torch.optim import Adam, AdamW, SGD

def get_optimizer(model, args):
    # optimizer�� �� �Ķ���� �� �־������
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == 'adamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ��� parameter���� grad���� 0���� �ʱ�ȭ
    optimizer.zero_grad()

    return optimizer