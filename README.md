# FunduSegmenter









Our code was verified on a single NVIDIA GeForce RTX 5090 GPU and AMD Ryzen 9 9950X CPU.

---
## 1. Installation
Create environment
```
conda create -n fundusegmenter python=3.11.0 -y
conda activate fundusegmenter
```
Install code and packages
```
git clone https://github.com/JusticeZzy/FunduSegmenter
cd FunduSegmenter
pip install -r requirements.txt
```

---
## 2. Download
Download RETFound pre-trained weights ```RETFound_mae_natureCFP.pth``` from [RETFound](https://github.com/rmaphoh/RETFound), and save it in ```FunduSegmenter```.

Download datasets [IDRiD](https://www.mdpi.com/2306-5729/3/3/25), [Drishti-GS](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2015/Arunava2015AComprehensive.pdf), [RIM-ONE-r3](https://ieeexplore.ieee.org/abstract/document/5999143?casa_token=R9T_bTVvDoMAAAAA:r2ipTjpfnGSzeUuqMIHDOrxI_T3XEeG67yP_cWiiwD2c9Xsom2CTBSLZXVswBow7BRDI_95VOt3cYw), and [REFUGE](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301100?casa_token=H1RvPw0rvRgAAAAA:XqD9RTnnyZ8dOg8Z9Wo54s16LRP-nxhfmhotHMMEugyYtt5hYQhHHcHkA18b0OnOhO7iSgJ2kmo).

You can directly download the processed datasets from [here](https://huggingface.co/JusticeZzy/FunduSegmenter) (Datasets.zip) and skip step 3. (Our private dataset [GoDARTS](https://academic.oup.com/ije/article/47/2/380/4107246) is not included.)

Our processed domain datasets for domain generalization baselines are in ```./domain_datasets/baseline_set```, for our models are in ```./domain_datasets/FunduSegmenter_set```.

---
## 3. Dataset preparation
Follow this step if you would like to use your own datasets. You may need to modify the related code to match your datasets. We also provide the t-SNE data dirtibution analysis in ```t_sne.ipynb```.

#### 3.1. Format preparation

Convert the Drishti-GS and RIM-ONE-r3 ground truth format to the REFUGE ground truth format by running ```prepare_Drishti_GS.ipynb``` and ```prepare_RIM_ONE_r3.ipynb``` in ```offline_datasets_prepare```.

Example visualization:

<div align="center">
<table>
  <tr>
    <td align="center">
      <img width="205" height="175" alt="image_drishtiGS_002" src="https://github.com/user-attachments/assets/92a9157e-df20-4905-8800-3cb0ff5a2460" /><br>
      Drishti-GS image
    </td>
    <td align="center">
      <img width="205" height="175" alt="drishtiGS_002_ODsegSoftmap" src="https://github.com/user-attachments/assets/fd1464d7-aae3-43a3-9408-5e53715ff438" /><br>
      OD ground truth
    </td>
    <td align="center">
      <img width="205" height="175" alt="drishtiGS_002_cupsegSoftmap" src="https://github.com/user-attachments/assets/90005290-d821-4c7b-97be-fb81e8706de0" /><br>
      OC ground truth
    </td>
    <td align="center">
      <img width="205" height="175" alt="new_mask_drishtiGS_002" src="https://github.com/user-attachments/assets/6f67272e-84b7-41e0-88e6-1d5969eb93ad" /><br>
      New ground truth
    </td>
  </tr>
</table>
</div>

<div align="center">
<table>
  <tr>
    <td align="center">
      <img width="214" height="142" alt="image_G-1-L" src="https://github.com/user-attachments/assets/d226035b-978a-4442-946d-cadd38eeede0" /><br>
      RIM-ONE-r3 image
    </td>
    <td align="center">
      <img width="107" height="142" alt="new_G-1-L" src="https://github.com/user-attachments/assets/bbcb08be-07e3-4910-a510-867402e232b8" /><br>
      New image     
    </td>
    <td align="center">
      <img width="214" height="142" alt="G-1-L-Disc-Avg" src="https://github.com/user-attachments/assets/be13e5fa-d7c7-409c-876b-d34dddef8ee8" /><br>
      OD ground truth
    </td>
    <td align="center">
      <img width="214" height="142" alt="G-1-L-Cup-Avg" src="https://github.com/user-attachments/assets/7b1540d7-e05e-4749-8125-185f1a613073" /><br>
      OC ground truth
    </td>
    <td align="center">
      <img width="107" height="142" alt="G-1-L" src="https://github.com/user-attachments/assets/453fb957-9509-4f4d-b53e-6c2698e0d9aa" /><br>
      New ground truth
    </td>
  </tr>
</table>
</div>

#### 3.2. OD centre cropping
Download the pre-trained [DUNet](https://www.sciencedirect.com/science/article/abs/pii/S0169260721000444?casa_token=19OZFuiKaQsAAAAA:3qza9yDwFd8qhnzq_73CReq3HQ5rjWV6Xv5f_6MNsBJceS_72dyjsg_pXieKBss2iLLuWl7qQJg) weight from [here](https://huggingface.co/JusticeZzy/FunduSegmenter) (DUNet_OD_CentreCrop_pretrained.pth). We pre-trained it on Drishti-GS, RIM-ONE-r3, REFUGE training, and REFUGE validation. DUNet is one of the baseline models. You can train your own weight by following the step 4. Alternatively, you can use any other reliable pre-trained weights to apply OD centre cropping pre-processing.

Run ```OD_centrecrop.ipynb``` in ```offline_datasets_prepare``` to apply OD centre cropping pre-processing by the pre-trained DUNet.

(Optional) You can test the Dice score of the pre-trained DUNet on your datasets if they contain OD ground truth. Run the following code:
```
python test_OD_centrecrop.py \
  --test_image_path (your own image dir) \
  --test_mask_path (your own ground truth dir) \
  --checkpoint_path ./DUNet_OD_centreCrop_pretrained.pth \
  --label_n_cls 3    # Important! 2 for OD only ground truth or 3 for OD/OC ground truth.
```

---
## 4. Train
Run the following code to train our ```FunduSegmenter```. Set ```--model_selection``` to ```baseline_RETFoundSegmenter```, ```baseline_DUNet```, ```baseline_TransUNet``` to train baselines (RETFoundSegmenter, DUNet, [TransUNet](https://github.com/Beckschen/TransUNet)) respectively. Note that we only copy the model architecture of TransUNet, and train it under our experimental pipeline. If you would like to reproduce results by running official code, click [here](https://github.com/Beckschen/TransUNet).
```
python train.py \
  --epochs 10000 \
  --model_selection FunduSegmenter \
  --output_channel (2 for OD only segmentation, 3 for OD/OC segmentation) \
  --label_n_cls (2 for OD only ground truth, 3 for OD/OC ground truth. You can set output_channel 2 and label_n_cls 3 to train OD only model if groundtruth contains OD/OC like REFUGE.) \
  --seed 112316 \
  --train_image_path (your own training ROI of images dir) \
  --train_mask_path (your own training ROI of ground truth dir) \
  --separate_val (1 for separate validation dataset like REFUGE (training/validation/testing), 0 for training/testing only datasets) \
  --val_image_path (your own validation ROI of images dir, can be ignored if separate_val 0) \
  --val_mask_path (your own validation ROI of ground truth dir, can be ignored if separate_val 0) \
  --num_workers 4 \
  --image_size 256 \
  --transform_mode designed_transform
```
You can run ```tensorboard --logdir=results/tensorboard``` to supervise the training progress.

---
## 5. Test
Run the following code to produce results if reconstructing maps from cropped ROIs to original size:
```
python test.py  \
  --label_n_cls (2 for OD only ground truth, 3 for OD/OC ground truth) \
  --test_image_path (your own testing ROI of images dir) \
  --test_original_mask_path (your own original testing ground truth dir) \
  --test_cropped_mask_path (your own testing ROI of ground truth dir) \
  --centre_file_path (your own centre file path, produced in step 3.2) \
  --image_size 256 \
  --model_selection FunduSegmenter \
  --checkpoint_path ./results/saved_weights/best_weights.pth \
  --output_channel (2 for OD only segmentation, 3 for OD/OC segmentation)
  --is_idrid    # Important! Disable for other datasets, or enable for IDRiD, since IDRiD was cropped to 1200×1200 ROIs.
```
Run the following code to produce results if directly reconstructing maps (domain generalization task or training with original images):
```
python test_nopadding.py  \
  --label_n_cls (2 for OD only ground truth, 3 for OD/OC ground truth) \
  --test_image_path (your own testing ROI of images dir) \
  --test_mask_path (your own testing ROI of ground truth dir) \
  --image_size 256 \
  --checkpoint_path ./results/saved_weights/best_weights.pth \
  --output_channel (2 for OD only segmentation, 3 for OD/OC segmentation)
```
The output segmentation maps are saved in ```./results/segmentation_map```.

---
## 6. Evaluation only
If you would like to use our trained weight to produce segmentation maps, you can follow the step 3 to pre-process your images, and then follow the step 5 (run ```test.py```) to produce segmentation maps. We provide a pre-trained FunduSegmenter which was trained on Drishti-GS, RIM-ONE-r3, REFUGE training, and REFUGE validation. The weight is available [here](https://huggingface.co/JusticeZzy/FunduSegmenter) (FunduSegmenter_CroppedImage.pth).

We also provide a pre-trained FunduSegmenter using original images which was also trained on Drishti-GS, RIM-ONE-r3, REFUGE training, and REFUGE validation. You can directly use the original images (need to follow step 3.1 to convert the ground truth format) and follow the step 5 (run ```test_nopadding.py```) to produce segmentation maps. The weight is available [here](https://huggingface.co/JusticeZzy/FunduSegmenter) (FunduSegmenter_OriginalImage.pth). Note that the performance of this weight is not widely verificated, so it could be unstable.

Additionally, all the weights trained and reported in our paper are available [here](https://huggingface.co/JusticeZzy/FunduSegmenter) (Weights.zip).

---
## 7. Baselines
#### 7.1. DUNet and TransUNet
You can implement DUNet and TransUNet by setting ```--model_selection``` to ```baseline_DUNet``` and ```baseline_TransUNet``` through ```train.py```. The official repository of TransUNet is [here](https://github.com/Beckschen/TransUNet).

You need to download the pre-trained weight ```imagenet21k_R50+ViT-B_16.npz``` from the official repository of TransUNet if implementing it under our pipeline.

#### 7.2. nnU-Net
You can implement nnU-Net by following the official [repository](https://github.com/MIC-DKFZ/nnUNet). Note that we implemented nnU-Net V2 in our paper.

We provide the required data converting files in ```./baselines_requirements/nnU-Net```. You can create your own code by following the official repository. We also provide the testing files in the folder to applying the metrics we used in our paper.

There is one necessary change. In ```./training/nnUNetTrainer/nnUNetTrainer.py```, you need to replace the following
```
line 579    splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
line 598    rnd = np.random.RandomState(seed=12345 + self.fold)
```
with
```
line 579    splits = generate_crossval_split(all_keys_sorted, seed=112316, n_splits=5)
line 598    rnd = np.random.RandomState(seed=112316 + self.fold)
```

#### 7.3. DoFE
You can implement DoFE by following the official [repository](https://github.com/emma-sjwang/Dofe).

There are some necessary changes.

1. If you use our processed datasets, you need to replace ```self.flags_DGS = ['gd', 'nd']``` with ```self.flags_DGS = ['dr']``` in ```line 40``` of ```./dataloaders/fundus_dataloader.py```. Ignore if not.

2. In ```line 45``` of ```./dataloaders/fundus_dataloader.py```, replace ```SEED = 1212``` with ```SEED = 112316```.

3. In ```line 78``` of ```./train.py```, replace ```torch.cuda.manual_seed(1337)``` with
   ```
   import random
   import numpy as np
   random.seed(112316)
   np.random.seed(112316)  
   torch.manual_seed(112316)
   torch.cuda.manual_seed_all(112316)
   ```

4. (Important!) The MobileNet part of DoFE load a pre-trained weight. However, the link of the weight is invalid. You need to download the same pre-trained weight from [here](https://huggingface.co/JusticeZzy/FunduSegmenter) (mobilenet_v2-6a65762b.pth) which is uploaded by us, and save it in ```./mobilenet_v2-6a65762b.pth```. In ```line 124``` of ```/networks/backbone/mobilenet.py```, replace ```pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')``` with ```pretrain_dict = torch.load('./mobilenet_v2-6a65762b.pth')```.

#### 7.4. RAM-DSIR
You can implement RAM-DSIR by following the official [repository](https://github.com/zzzqzhou/RAM-DSIR).

There are some necessary changes.

1. If you use our processed datasets, you need to replace the dataset list files with ours in ```./baselines_requirements/RAM-DSIR```. Ignore if not.

2. In ```./code/dataset/fundus.py```, replace the following
   ```
   line 78     with open(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'test.list'), 'r') as f:
   line 93     img = Image.open(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[0]))
   line 97     mask = Image.open(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[1])).convert('L')
   line 206    with open(os.path.join(self.base_dir, other_domain_name, 'train.list'), 'r') as f:
   line 208    other_id = np.random.choice(other_id_path).replace('\n', '').split(' ')[0]
   ```
   with
   ```
   line 78     with open(os.path.join(self.base_dir, self.domain_name[self.domain_idx]+'_test.list'), 'r') as f:
   line 93     img = Image.open(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[0][8:]))
   line 97     mask = Image.open(os.path.join(self.base_dir, cur_domain_name, id.split(' ')[1][8:])).convert('L')
   line 206    with open(os.path.join(self.base_dir, other_domain_name+'_train.list'), 'r') as f:
   line 208    other_id = np.random.choice(other_id_path).replace('\n', '').split(' ')[0][8:]
   ```
3. In ```line 60``` of ```./code/train.py```, replace ```parser.add_argument('--seed', type=int,  default=1337, help='random seed')``` with ```parser.add_argument('--seed', type=int,  default=112316, help='random seed')```.

#### 7.5. TVConv
You can implement TVConv by following the official [repository](https://github.com/JierunChen/TVConv).

There are some necessary changes.

1. If you use our processed datasets, you need to replace ```self.flags_DGS = ['gd', 'nd']``` with ```self.flags_DGS = ['dr']``` in ```line 40``` of ```./od_oc_segmentation/dataloaders/fundus_dataloader.py```. Ignore if not.

2. In ```line 45``` of ```./od_oc_segmentation/dataloaders/fundus_dataloader.py```, replace ```SEED = 1212``` with ```SEED = 112316```.

3. In ```line 41``` of ```./od_oc_segmentation/odoc_train_test.py```, replace ```torch.cuda.manual_seed(1337)``` with
   ```
   import random
   import numpy as np
   random.seed(112316)
   np.random.seed(112316)  
   torch.manual_seed(112316)
   torch.cuda.manual_seed_all(112316)
   ```

---
## 8. Acknowledgments
Our work is benefited from [RETFound](https://github.com/rmaphoh/RETFound) and [Segmenter](https://github.com/rstrudel/segmenter). Thanks for their great work!

---
## 9. Citation
If you find our work useful, please consider citing
```
```
