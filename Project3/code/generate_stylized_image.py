import torch
from torchvision import transforms

import utils
from transformer_net import TransformerNet

content_scale = None


def generate_stylized_image(args):
    """
    Creates stylized image based on the arguments passed
    :param args: (content_image, style_image, model, etc.)
    :return: void
    """
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet(style_number=args.style_num)
        state_dict = torch.load(args.model)
        style_model.load_state_dict(state_dict, strict=False)
        style_model.to(device)
        output = style_model(content_image, style_id = [args.style_id]).cpu()

    utils.save_image('output/'+args.output_image+'_style'+str(args.style_id)+'.jpg', output[0])