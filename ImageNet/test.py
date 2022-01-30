
ini_lr=0.1
warmup=5
minestone0 = [warmup]
minestone1 = [30, 100, 170, 240]  # 0.01
minestone2 = [60, 130, 200, 270]  # 0.001
minestone3 = [90, 160, 230, 300]  # 0.0001
for epoch in range(0, 310):
    # lr_scheduler(optimizer, epoch)
    if epoch < 5:
        lr = ini_lr * (epoch + 1) / (warmup + 1)
    elif epoch in minestone0:
        lr = 0.1
    elif epoch in minestone1:
        lr = 0.01
        if epoch != 30:
            best_prec1 = 0

    elif epoch in minestone2:
        lr = 0.001
    elif epoch in minestone3:
        lr = 0.0001

    param_group = lr
    print(f'current learning rate is {lr}')