from torch.optim import Adam, AdamW, SGD

def get_optimizer(model, args):
    # optimizer에 모델 파라미터 꼭 넣어줘야함
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == 'adamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()

    return optimizer