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
from models.cycle_gan_model import CycleGANModel
from util.object import Object
from util.save_images import save_images

if __name__ == '__main__':
    options = Object(**dict(
        aspect_ratio=1.0,
        batch_size=1,
        beta1=0.5,
        checkpoints_dir='/content/cyclegan/checkpoints',
        continue_train=False,
        crop_size=256,
        dataroot='/content/cyclegan/datasets/monet2photo',
        direction='AtoB',
        display_freq=400,
        display_id=-1,
        display_winsize=256,
        epoch='latest',
        epoch_count=1,
        eval=False,
        gan_mode='lsgan',
        gpu_ids=[0],
        init_gain=0.02,
        init_type='normal',
        input_nc=3,
        isTrain=True,
        lambda_A=10.0,
        lambda_B=10.0,
        lambda_identity=0.5,
        load_iter=0,
        load_size=286,
        lr=0.0002,
        lr_policy='linear',
        max_dataset_size=float('inf'),
        n_layers_D=3,
        name='bleus1',
        ndf=64,
        netD='basic',
        netG='resnet_9blocks',
        ngf=64,
        niter=1,
        niter_decay=0,
        no_dropout=True,
        norm='instance',
        num_threads=0,
        phase='test',
        pool_size=50,
        preprocess='resize_and_crop',
        print_freq=100,
        results_dir='/content/cyclegan/results',
        save_by_iter=False,
        save_epoch_freq=5,
        save_latest_freq=5000,
        serial_batches=True,
        no_flip=True,
        num_test=10,
        num_threads=4,
        output_nc=3,
        update_html_freq=1000,
        verbose=False,
    ))

    dataset = CustomDatasetDataLoader(options)

    model = CycleGANModel(options)
    model.setup(options)

    # create a website
    web_dir = os.path.join(options.results_dir, options.name, '%s_%s' % (options.phase, options.epoch))  # define the website directory
    img_dir = os.path.join(web_dir, 'images')
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

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

    # webpage.save()  # save the HTML
