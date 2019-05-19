"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.test_model import TestModel
from util.object import Object
from util.save_images import save_images


def run(options_raw):
    options = Object(**options_raw)

    dataset = CustomDatasetDataLoader(options)

    model = TestModel(options)
    model.setup(options)

    img_dir = os.path.join(options.results_dir, options.name)

    for i, data in enumerate(dataset):
        if i >= options.max_image_iterations:
            break

        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(img_dir, visuals, img_path)

if __name__ == '__main__':
    options_raw = dict(
        batch_size=1,
        checkpoints_dir='./checkpoints',
        crop_size=256,
        dataroot='./datasets/test',
        gpu_ids=[0],
        init_gain=0.02,
        init_type='normal',
        input_nc=3,
        isTrain=False,
        load_size=256,
        n_layers_D=3,
        name='bleus_simple',
        ndf=64,
        ngf=64,
        no_flip=True,
        norm='instance',
        max_image_iterations=50,
        num_threads=0,
        output_nc=3,
        preprocess='resize_and_crop',
        results_dir='./results',
        use_dropout=False,
        verbose=False,
    )

    run(options_raw)
