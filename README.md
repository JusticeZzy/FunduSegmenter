# FunduSegmenter










### 1. Installation
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
### 2. Download
Download RETFound pre-trained weights ```RETFound_mae_natureCFP.pth``` from [RETFound](https://github.com/rmaphoh/RETFound), and save it in ```FunduSegmenter```.

Download datasets [IDRID](https://www.mdpi.com/2306-5729/3/3/25), [Drishti-GS](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2015/Arunava2015AComprehensive.pdf), [RIM-ONE-r3](https://ieeexplore.ieee.org/abstract/document/5999143?casa_token=R9T_bTVvDoMAAAAA:r2ipTjpfnGSzeUuqMIHDOrxI_T3XEeG67yP_cWiiwD2c9Xsom2CTBSLZXVswBow7BRDI_95VOt3cYw), and [REFUGE](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301100?casa_token=H1RvPw0rvRgAAAAA:XqD9RTnnyZ8dOg8Z9Wo54s16LRP-nxhfmhotHMMEugyYtt5hYQhHHcHkA18b0OnOhO7iSgJ2kmo).

(Optional) You can directly download the processed datasets from [here]() and skip step 3. (Our private dataset [GoDARTS](https://academic.oup.com/ije/article/47/2/380/4107246) is not included.)xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

### (Optional) 3. Dataset preparation
Follow this step if you would like to use your own datasets. You may need to modify the related code to match your datasets.

#### 3.1. Format preparation

Convert the Drishti-GS and RIM-ONE-r3 mask format to the REFUGE mask format by running ```prepare_Drishti_GS.ipynb``` and ```prepare_RIM_ONE_r3.ipynb``` in ```offline_datasets_prepare```.

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
      Original OD mask
    </td>
    <td align="center">
      <img width="205" height="175" alt="drishtiGS_002_cupsegSoftmap" src="https://github.com/user-attachments/assets/90005290-d821-4c7b-97be-fb81e8706de0" /><br>
      Original OC mask
    </td>
    <td align="center">
      <img width="205" height="175" alt="new_mask_drishtiGS_002" src="https://github.com/user-attachments/assets/6f67272e-84b7-41e0-88e6-1d5969eb93ad" /><br>
      New mask
    </td>
  </tr>
</table>
</div>

<div align="center">
<table>
  <tr>
    <td align="center">
      <img width="214" height="142" alt="image_G-1-L" src="https://github.com/user-attachments/assets/d226035b-978a-4442-946d-cadd38eeede0" /><br>
      RIM-ONE-r3 original image
    </td>
    <td align="center">
      <img width="107" height="142" alt="new_G-1-L" src="https://github.com/user-attachments/assets/bbcb08be-07e3-4910-a510-867402e232b8" /><br>
      new image
    </td>
    <td align="center">
      <img width="214" height="142" alt="G-1-L-Disc-Avg" src="https://github.com/user-attachments/assets/be13e5fa-d7c7-409c-876b-d34dddef8ee8" /><br>
   Original OD mask
    </td>
    <td align="center">
      <img width="214" height="142" alt="G-1-L-Cup-Avg" src="https://github.com/user-attachments/assets/7b1540d7-e05e-4749-8125-185f1a613073" /><br>
      Original OC mask
    </td>
    <td align="center">
      <img width="107" height="142" alt="G-1-L" src="https://github.com/user-attachments/assets/453fb957-9509-4f4d-b53e-6c2698e0d9aa" /><br>
      New mask
    </td>
  </tr>
</table>
</div>

#### 3.2. OD centre cropping
Download the pre-trained [DUNet](https://www.sciencedirect.com/science/article/abs/pii/S0169260721000444?casa_token=19OZFuiKaQsAAAAA:3qza9yDwFd8qhnzq_73CReq3HQ5rjWV6Xv5f_6MNsBJceS_72dyjsg_pXieKBss2iLLuWl7qQJg) weight from [here]()xxxxxxxxxxxxxxxxxxxxxxxxxxx. We pre-trained it on Drishti-GS, RIM-ONE-r3, REFUGE training, and REFUGE validation. DUNet is one of the baseline models. You can train your own weight by following the step 4. Alternatively, you can use any other reliable pre-trained weights to apply OD centre cropping pre-processing.

Run ```OD_centrecrop.ipynb``` in ```offline_datasets_prepare``` to apply OD centre cropping pre-processing by the pre-trained DUNet.

(Optional) You can test the Dice score of the pre-trained DUNet on your datasets if they contain OD groundtruth. Run the following code:
```
python test_OD_centrecrop.py \
  --test_image_path (your own image dir) \
  --test_mask_path (your own mask dir) \
  --checkpoint_path ./DUNet_OD_centreCrop_pretrained.pth \
  --label_n_cls 3    # Important! 2 for OD only masks or 3 for OD/OC masks.
```

### 4. Train
Run the following code to train our ```FunduSegmenter```. Set model_selection to ```baseline_RETFoundSegmenter```, ```baseline_DUNet```, ```baseline_TransUNet``` to train baselines (RETFoundSegmenter, DUNet, [TransUNet](https://github.com/Beckschen/TransUNet)) respectively. Note that we only copy the model architecture of TransUNet, and train it under our experimental pipeline. If you would like to reproduce results by running official code, click [here](https://github.com/Beckschen/TransUNet).
```
python train.py \
  --epochs 10000 \
  --model_selection FunduSegmenter \
  --output_channel (2 for OD only segmentation, 3 for OD/OC segmentation) \
  --label_n_cls (2 for OD only groundtruth, 3 for OD/OC masks. You can set output_channel 2 and label_n_cls 3 to train OD only model if groundtruth contains OD/OC like REFUGE.) \
  --seed 112316 \
  --train_image_path (your own training image dir) \
  --train_mask_path (your own training mask dir) \
  --separate_val (1 for separate validation dataset like REFUGE (training/validation/testing), 0 for training/testing only datasets) \
  --val_image_path (your own validation image dir, can be ignored if separate_val 0) \
  --val_mask_path (your own validation mask dir, can be ignored if separate_val 0) \
  --num_workers 4 \
  --image_size 256 \
  --transform_mode designed_transform
```
You can run ```tensorboard --logdir=results/tensorboard``` to supervise the training process.
