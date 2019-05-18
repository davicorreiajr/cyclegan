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

if __name__ == '__main__':
    options = Object(**dict(
        aspect_ratio=1.0,
        batch_size=1,#
        checkpoints_dir='/content/cyclegan/checkpoints',
        crop_size=256,#
        dataroot='/content/cyclegan/datasets/test',#
        dataset_mode='single',
        # direction='AtoB',
        display_winsize=256,
        epoch='latest',
        eval=False,
        gpu_ids=[],
        init_gain=0.02,
        init_type='normal',
        input_nc=3,#
        isTrain=False,
        load_iter=0,
        load_size=256,#
        # max_dataset_size=float('inf'),
        model='test',
        model_suffix='',
        n_layers_D=3,
        name='style_vangogh_pretrained',
        ndf=64,
        netD='basic',
        netG='resnet_9blocks',
        ngf=64,
        no_flip=True,#
        norm='instance',
        ntest=float('inf'),
        num_test=50,
        num_threads=0,#
        output_nc=3,
        # phase='test',
        preprocess='resize_and_crop',#
        results_dir='/content/cyclegan/results',
        serial_batches=True,#
        suffix='',
        verbose=False,
        use_dropout=False,
    ))

    dataset = CustomDatasetDataLoader(options)

    model = TestModel(options)
    model.setup(options)

    img_dir = os.path.join(options.results_dir, options.name)

    if options.eval:
        model.eval()

    for i, data in enumerate(dataset):
        if i >= options.num_test:
            break

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(img_dir, visuals, img_path, aspect_ratio=options.aspect_ratio, width=options.display_winsize)
