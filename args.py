class args():
    epochs = 16
    batch_size = 4

    train_ir = 'E:/clip picture last/ir/'
    train_vi = 'E:/clip picture last/vi/'

    hight = 256
    width = 256

    save_model_dir = "models_training"
    save_loss_dir = "loss"

    weight_SSIM = 10
    weight_Grad = 16
    weight_Intensity = 14

    r = 0.4

    cuda = 1

    g_lr = 0.0001
    d_lr = 0.0004
    log_interval = 5
    log_iter = 1

