# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
from monai.inferers import sliding_window_inference

from utils.data_utils import get_loader
from trainer import dice
import argparse
import vnet
import torch.nn as nn
import nibabel as nib

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./runs/logic_rerangeintensity_500/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='./dataset/dataset1/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_1.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='model.pt', type=str, help='pretrained model name')
parser.add_argument('--saved_checkpoint', default='ckpt', type=str, help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=16, type=int, help='feature size dimention')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=4, type=int, help='number of output channels')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--a_min', default=0.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=255.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')

parser.add_argument('--use_normal_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')

parser.add_argument('--dice', action='store_true')

def main():
    def save_nifti(data,save_path):
        data_ = nib.load("/home/maoyuejingxian/UNETR_semicon/dataset/dataset0/imagesTs/zdb_memory_d1_b2_s1_2_void.nii.gz")
        header = data_.header
        nifti_image = nib.Nifti1Image(data,None,header)
        nib.save(nifti_image,"/home/maoyuejingxian/vnet_visualization/logic/"+save_path)
        print('save file sucess')

    args = parser.parse_args()
    args.test_mode = True

    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    nll = True
    if args.dice:
        nll = False

    if args.saved_checkpoint == 'torchscript':
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == 'ckpt':
        model = vnet.VNet(elu=False, nll=nll)
        model_dict = torch.load(pretrained_pth)
        model.load_state_dict(model_dict['state_dict'])
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        dice1 = []
        dice2 = []
        dice3 = []
        dice4 = []
        for i, batch in enumerate(val_loader):
            # print(type(np.unique(batch["label"])[0]))
            # print(np.unique(batch["label"]))
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

            # test_val_labels = val_labels.cpu().numpy()
            # test_val_labels2 = batch['label']

            # # print(type(np.unique(test_val_labels)[0]))
            # print("***************************************************")
            # print(np.unique(test_val_labels))


            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs,
                                                   (96, 96, 96),
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            # print(val_labels.shape)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :].round().astype(np.uint8)
            # print(val_outputs.shape)
            # print(val_labels.shape)

            # print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            # print(np.unique(val_labels))
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # print(np.unique(val_outputs))


            # organ_Dice = dice(val_outputs[0] == 1, val_labels[0] == 1)
            image_data = val_outputs.reshape(96,96,96)
            save_nifti(image_data,img_name)

            dice_list_sub = []
            for i in range(0, 4):

                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                # print("dice for {}".format(i))
                # print(organ_Dice)
                # dice_list_sub.append(organ_Dice)
            
                if i!=0:
                    print("dice for {}".format(i))
                    print(organ_Dice)
                    dice_list_sub.append(organ_Dice)

                    if i == 1:
                        dice1.append(organ_Dice)

                    if i == 2:
                        dice2.append(organ_Dice)

                    if i == 3:
                        dice3.append(organ_Dice)

                    if i == 4:
                        dice4.append(organ_Dice)

            mean_dice = np.mean(dice_list_sub)
            print("Mean Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)

            mean_dice1 = np.mean(dice1)
            mean_dice2 = np.mean(dice2)
            mean_dice3 = np.mean(dice3).round(4)
            mean_dice4 = np.mean(dice4)
        print("Dice1_average: {}".format(mean_dice1))
        print("Dice2_average: {}".format(mean_dice2))
        print("Dice3_average: {}".format(mean_dice3))
        print("Dice4_average: {}".format(mean_dice4))

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

if __name__ == '__main__':
    main()