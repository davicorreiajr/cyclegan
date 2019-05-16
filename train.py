from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.cycle_gan_model import CycleGANModel

if __name__ == '__main__':
    options = dict(
        crop_size=256,
        dataroot='datasets/monet2photo',
        load_size=286,
        phase='train',
        preprocess='resize_and_crop',
        no_flip=False,
        num_threads=4,
    )

    dataset = CustomDatasetDataLoader(options)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = CycleGANModel(options)
    print("model [%s] was created" % type(model).__name__)
