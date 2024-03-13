import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from cyclegan.train import Generator
from glob import glob
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    device = torch.device('cpu')
    netG_A2B = Generator(3, 3)
    netG_B2A = Generator(3, 3)

    model_path = f'resources/gan_pic'

    netG_A2B.load_state_dict(torch.load(f'{model_path}/netG_A2B.pth',map_location='cpu'))
    netG_B2A.load_state_dict(torch.load(f'{model_path}/netG_B2A.pth',map_location='cpu'))

    # Set model's test mode
    netG_A2B.eval()
    netG_B2A.eval()

    transforms_ = [transforms.Resize(int(256 * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(256),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transforms_)

    for file in glob('jpg/*'):
        real_A = transform(Image.open(file))
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        save_image(fake_B, f'jpg/{os.path.splitext(os.path.split(file)[1])[0]}_AI.png')

    exit()
    file1 = '/data1/tmp_data/植物书法数据集/和平型/金.jpg'
    file2 = '/data1/tmp_data/植物书法数据集/和平型（AI）/金.jpeg'

    real_A = transform(Image.open(file1))
    real_B = transform(Image.open(file2).convert('RGB'))

    # Generate output
    fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(fake_A, 'outputA.png')
    save_image(fake_B, 'outputB.png')