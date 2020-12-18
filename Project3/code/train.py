import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg import Vgg16

content_weight = 1e5
style_weight = 1e10
learning_rate = 1e-3
log_interval = 25
checkpoint_interval = 100


def train(args):
    """
    Trains the models
    :param args: parameters
    :return: saves the model and checkpoints
    """
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_transform = transforms.Compose([
        transforms.Resize(args.image_size),  # the shorter side is resize to match image_size
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),  # to tensor [0,1]
        transforms.Lambda(lambda x: x.mul(255))  # convert back to [0, 255]
    ])
    train_dataset = datasets.ImageFolder(args.dataset, image_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # to provide a batch loader

    style_image = [f for f in os.listdir(args.style_image)]
    style_num = len(style_image)
    print(style_num)

    transformer = TransformerNet(style_number=style_num).to(device)
    adam_optimizer = Adam(transformer.parameters(), learning_rate)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.Resize(args.style_size),
        transforms.CenterCrop(args.style_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    style_batch = []

    for i in range(style_num):
        if ".ipynb" not in style_image[i]:
            style = utils.load_image(args.style_image + style_image[i], size=args.style_size)
            style = style_transform(style)
            print(style.shape, style_image[i])
            style_batch.append(style)

    style = torch.stack(style_batch).to(device)
    # print("After stack")
    features_style = vgg(utils.normalize_batch(style))
    # print("After feature style")
    gram_style = [utils.gram_matrix(y) for y in features_style]
    # print("starting epochs")
    for e in range(args.epochs):
        with open('/home/sbanda/Fall20-DL-CG/Project3/log.txt', 'a') as reader:
            reader.write("Epoch " + str(e) + ":->\n")
        transformer.train()
        aggregate_content_loss = 0.
        aggregate_style_loss = 0.
        counter = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            print(batch_id)
            if n_batch < args.batch_size:
                break

            counter += n_batch
            # Initialize gradients to zero
            adam_optimizer.zero_grad()

            batch_style_id = [i % style_num for i in range(counter - n_batch, counter)]
            y = transformer(x.to(device), style_id=batch_style_id)

            x = utils.normalize_batch(x)
            y = utils.normalize_batch(y)

            features_x = vgg(x.to(device))
            features_y = vgg(y.to(device))

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for feature_y, gm_style in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(feature_y)
                style_loss += mse_loss(gm_y, gm_style[batch_style_id, :, :])

            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            adam_optimizer.step()

            aggregate_content_loss += content_loss.item()
            aggregate_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, counter, len(train_dataset),
                                  aggregate_content_loss / (batch_id + 1),
                                  aggregate_style_loss / (batch_id + 1),
                                  (aggregate_content_loss + aggregate_style_loss) / (batch_id + 1)
                )
                with open('/home/sbanda/Fall20-DL-CG/Project3/log.txt', 'a') as reader:
                    reader.write(mesg + "\n")
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':',
                                                                                                          '') + "_" + str(
        int(
            content_weight)) + "_" + str(int(style_weight)) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)
