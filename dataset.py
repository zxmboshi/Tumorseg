import os
import numpy as np
import nibabel as nib
import mindspore
from mindspore.dataset import GeneratorDataset

def load_nifti_image(filepath,dtype):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    pathname = filepath.split('/')
    name = pathname[-2]
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_fdata(),dtype = dtype)
    out_nii_array = np.squeeze(out_nii_array)
    meta={'affine':nim.affine,
          'dim':nim.header['dim'],
          'pixdim':nim.header['pixdim'],
          'name':name,
          'lastname':os.path.basename(filepath)
        }
    return out_nii_array,meta

class NIFTIdataset(object):
    def __init__(self, root_dir, split, transform=None):
        super(NIFTIdataset,self).__init__()
        
        self.split = split
        self.dir = os.path.join(root_dir, split)
        self.filenames = sorted([os.path.join(self.dir, x) for x in os.listdir(self.dir) ])
        self.transform = transform
        print('Number of {0} images: ::{1} NIFTIs'.format(split,self.__len__()))

    def __getitem__(self, index):
        image_filenames = os.path.join(self.filenames[index],'imaging.nii.gz')
        label_filenames = os.path.join(self.filenames[index],'segmentation.nii.gz')
        image, image_meta = load_nifti_image(image_filenames, dtype=np.float32)
        label, label_meta = load_nifti_image(label_filenames, dtype=np.float32)
        
        max_value = np.max(image)
        min_value = np.min(image)
        clip_min = min_value + 0.005 * (max_value-min_value)
        clip_max = max_value - 0.005 * (max_value-min_value)
        image = np.clip(image, clip_min, clip_max)

        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean)/std
        
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        return image, label

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    dataset_test = NIFTIdataset('../../dataset2', split = 'test')
    dataset_test = GeneratorDataset(source=dataset_test, column_names=["image", "label"])
    dataset_test = dataset_test.batch(1)
    for data in dataset_test.create_dict_iterator():
        print(data["image"].shape, data["label"].shape)


