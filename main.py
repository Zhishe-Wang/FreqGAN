import time
from train import train
from Modules import generate
from args import args
import utils
import torch
import os
from Net import Generator_DWT

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# flag = 1
flag = 0

if flag == 1:
    IS_TRAINING = True
else:
    IS_TRAINING = False


def load_model(model_path):
    G_model = Generator_DWT()
    G_model.load_state_dict(torch.load(model_path), False)
    print('# generator parameters:', sum(param.numel() for param in G_model.parameters()))
    G_model.eval()
    G_model.cuda()
    return G_model


def main():
    # training
    if IS_TRAINING:
        train_data_ir = utils.list_images(args.train_ir)
        train_data_vi = utils.list_images(args.train_vi)
        train(train_data_ir, train_data_vi)

    # testing
    else:
        print("\nBegin to generate pictures ...\n")

        model_name = 'FusionModel.model'

        test_imgs_path_ir = "./test_imgs/TNO/ir/"
        test_imgs_path_vi = "./test_imgs/TNO/vi/"

        print('Dateset begin to test.')
        model_path = os.path.join(os.getcwd(), 'models_training', model_name)
        with torch.no_grad():
            model = load_model(model_path)
            model.eval()
            model.cuda()
            begin = time.time()
            for i in range(1000, 1025):
                index = i + 1
                # TNO
                ir_path = test_imgs_path_ir + "IR" + str(index) + ".png"
                vis_path = test_imgs_path_vi + "VIS" + str(index) + ".png"

                generate(model, ir_path, vis_path, model_path, index, mode='L')
            end = time.time()
            print("consumption time of generating:%s " % (end - begin))
            print("consumption time of generating:%s " , (end - begin)/25)


if __name__ == "__main__":
    main()
