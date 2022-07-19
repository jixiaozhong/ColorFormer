import argparse
import cv2
import glob
import numpy as np
import os
import torch
import tqdm

from basicsr.archs.colorformer_arch import ColorFormer as models
from basicsr.data.transforms import rgb2lab
from basicsr.utils.img_process_util import color_postprocess, gamma_lut
from basicsr.utils.img_util import tensor_lab2rgb
from basicsr.data.val_dataset import ValDataset
from queue import Queue
import _thread

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

write_buffer = Queue(maxsize=500)
def clear_write_buffer(args, write_buffer):
    while True:
        item = write_buffer.get()
        for name in item.keys():
            cls, filename = os.path.split(name)
            if cls:
                os.makedirs(os.path.join(args.output, cls), exist_ok=True)
            cv2.imwrite(os.path.join(args.output, name), item[name])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        ''  # noqa: E501
    )
    parser.add_argument(
        '--input',
        type=str,
        default='',
        help='input test image folder or video path')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='output folder or video path')
    parser.add_argument('--input_size', type=int, default=256, help='input size')
    args = parser.parse_args()

    # set up model
    model = models('GLHTransformer', input_size=[args.input_size, args.input_size], num_output_channels=2, last_norm='Spectral', do_normalize=False)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)
    os.makedirs(args.output, exist_ok=True)
    dataset = ValDataset(args.input, args.input_size, True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

    _thread.start_new_thread(clear_write_buffer, (args, write_buffer))
    with torch.no_grad():
        for (imgs, img_l, names) in tqdm.tqdm(dataloader):
            imgs = imgs.to(device)
            img_l = img_l.to(device)
            outs = model(imgs)
            # outs = outs.cpu()
            outs = torch.cat([img_l, outs], dim=1)
            outs = tensor_lab2rgb(outs)
            outs = outs.cpu()
            for i in range(len(names)):
                output = outs[i].data.float().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output_img = (output * 255.0).round().astype(np.uint8)
                write_buffer.put({names[i]: output_img})
    import time
    while(not write_buffer.empty()):
        time.sleep(0.1)

if __name__ == '__main__':
    main()
