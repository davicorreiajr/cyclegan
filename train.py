import time
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from models.cycle_gan_model import CycleGANModel
from util.object import Object


def print_current_losses(epoch, iters, losses, t_comp, t_data):
    """print current losses on console

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message

if __name__ == '__main__':
    options = Object(**dict(
        batch_size=1,
        beta1=0.5,
        checkpoints_dir='/content/cyclegan/checkpoints',
        continue_train=False,
        crop_size=256,
        dataroot='/content/cyclegan/datasets/monet2photo',
        direction='AtoB',
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
        phase='train',
        pool_size=50,
        preprocess='resize_and_crop',
        print_freq=100,
        save_by_iter=False,
        save_epoch_freq=5,
        save_latest_freq=5000,
        serial_batches=False,
        no_flip=False,
        num_threads=4,
        output_nc=3,
        update_html_freq=1000,
        verbose=False,
    ))

    dataset = CustomDatasetDataLoader(options)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = CycleGANModel(options)
    print("model [%s] was created" % type(model).__name__)

    model.setup(options)
    total_iters = 0

    print('Starting the training...')
    for epoch in range(options.epoch_count, options.niter + options.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        print('\nRunning over the dataset...')
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if i > 50:
                continue

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % options.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            # visualizer.reset()
            total_iters += options.batch_size
            epoch_iter += options.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # if total_iters % options.display_freq == 0:   # display images on visdom and save images to a HTML file
            #     save_result = total_iters % options.update_html_freq == 0
            #     model.compute_visuals()
            #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # if total_iters % options.print_freq == 0:    # print training losses and save logging information to the disk
            #     losses = model.get_current_losses()
            #     t_comp = (time.time() - iter_start_time) / options.batch_size
            #     print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            #     if opt.display_id > 0:
            #         visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % options.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if options.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        losses = model.get_current_losses()
        t_comp = (time.time() - iter_start_time) / options.batch_size
        print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

        print('Saving network...')
        model.save_networks('latest')
        # if epoch % options.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, options.niter + options.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    print('\nFinish of training.')

