import multiprocessing
import shutil
from multiprocessing import Pool
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from skimage import img_as_ubyte


def load_and_convert_case(input_image: str, input_seg: str, output_image: str, output_seg: str):
    seg = io.imread(input_seg)  
    seg = img_as_ubyte(seg)
    seg[seg == 255] = 1

    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


if __name__ == "__main__":
   
    source = '/home/zhenyi/Project/nnUNet/nnUNet_IDRID/IDRID'

    dataset_name = 'Dataset001_IDRID'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, 'training')
    test_source = join(source, 'testing')

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(train_source, 'masks'), join=False, suffix='tif')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            image_filename = v[:-7] + '.jpg'
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(train_source, 'images', image_filename),
                         join(train_source, 'masks', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v[:-4] + '.png'),
                         
                     ),)
                )
            )

        # test set
        valid_ids = subfiles(join(test_source, 'masks'), join=False, suffix='tif')
        for v in valid_ids:
            image_filename = v[:-7] + '.jpg'
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(test_source, 'images', image_filename),
                         join(test_source, 'masks', v),
                         join(imagests, v[:-4] + '_0000.png'),
                         join(labelsts, v[:-4] + '.png'),
                         
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'od': 1},
                          num_train, '.png', dataset_name=dataset_name)