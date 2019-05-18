import time
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.cycle_gan_model import CycleGANModel
from util.object import Object


def print_current_losses(epoch, iters, losses):
    """print current losses on console

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, iters: %d) ' % (epoch, iters)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)


def run(options_raw):
    options = Object(**options_raw)
    print('####### Options ####### ')
    for key in list(options_raw.keys()):
        print('%s = %s' % (key, options_raw[key]))
    print('\n\n')

    dataset = CustomDatasetDataLoader(options)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = CycleGANModel(options)
    print("model [%s] was created" % type(model).__name__)

    model.setup(options)

    training_start_time = time.time()
    print('Starting the training...\n')
    for epoch in range(options.epoch_count, options.niter + options.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        print('Running over the dataset...')
        for i, data in enumerate(dataset):
            if i > options.max_image_iterations:
                continue

            epoch_iter += options.batch_size

            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()

            if i % options.print_freq == 0:
                print('Finish %d-th image' % i)

        losses = model.get_current_losses()
        print_current_losses(epoch, epoch_iter, losses)

        print('Saving network...')
        model.save_networks('latest')
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, options.niter + options.niter_decay, time.time() - epoch_start_time))
        # update learning rates at the end of every epoch.
        model.update_learning_rate()

    print('\nFinish of training in ', time.time() - training_start_time)


if __name__ == '__main__':
    options_raw = dict(
        batch_size=1,
        beta1=0.5,
        checkpoints_dir='./checkpoints',
        continue_train=False,
        crop_size=256,
        dataroot='./datasets/vangogh2photo',
        # direction='AtoB',
        display_freq=400,
        epoch='latest',
        epoch_count=1,
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
        max_image_iterations=800,
        max_dataset_size=float('inf'),
        n_layers_D=3,
        name='bleus',
        ndf=64,
        netD='basic',
        netG='resnet_9blocks',
        ngf=64,
        niter=5,
        niter_decay=5,
        no_flip=False,
        norm='instance',
        num_threads=4,
        output_nc=3,
        # phase='train',
        pool_size=50,
        preprocess='resize_and_crop',
        print_freq=100,
        save_by_iter=False,
        save_epoch_freq=5,
        save_latest_freq=5000,
        # serial_batches=False,
        update_html_freq=1000,
        use_dropout=False,
        verbose=False,
    )

    run(options_raw)

