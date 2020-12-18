from PIL import Image


def load_image(filename, size=None, scale=None):
    """
    Loads the image with given file name
    :param filename: image path
    :param size: Size of the image
    :param scale: scale factor of the image
    :return: transformed image
    """
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    """
    Saves the image after few transformations
    :param filename: image path
    :param data: used for clamping and cloning
    :return: void
    """
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    """
    Computes gram matrix of given input feature vector y
    :param y: style image
    :return: gram matrix
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    """
    Normalizing data before passing as input into VGG
    :param batch: batch of images
    :return: normalized batch
    """
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std