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
Download RETFound pre-trained weights 'RETFound_mae_natureCFP.pth' from [RETFound](https://github.com/rmaphoh/RETFound), and save it in FunduSegmenter.

Download Datasets [IDRID](https://www.mdpi.com/2306-5729/3/3/25), [Drishti-GS](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2015/Arunava2015AComprehensive.pdf), [RIM-ONE-r3](https://ieeexplore.ieee.org/abstract/document/5999143?casa_token=R9T_bTVvDoMAAAAA:r2ipTjpfnGSzeUuqMIHDOrxI_T3XEeG67yP_cWiiwD2c9Xsom2CTBSLZXVswBow7BRDI_95VOt3cYw), and [REFUGE](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301100?casa_token=H1RvPw0rvRgAAAAA:XqD9RTnnyZ8dOg8Z9Wo54s16LRP-nxhfmhotHMMEugyYtt5hYQhHHcHkA18b0OnOhO7iSgJ2kmo).

### 3. Datasets preparation
Process the Drishti-GS and RIM-ONE-r3 mask format to the REFUGE mask format by running ```prepare_Drishti_GS.ipynb``` and ```prepare_RIM_ONE_r3.ipynb``` in ```offline_datasets_prepare```.

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

### 4. Pre-processing
