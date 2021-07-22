from utils_ls import *

# apex
has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False


def main():
    # step 1. init logger file
    args = parse_args()
    logger = init_logger(log_path=args.log_path, logger_name="bisenet_logger", phrase='train')

    # step 2. configs update and review
    configs = merge_configs(None, args.project_config_path, args.default_config_path)

    # step 4. init random seeds
    fix_random_seeds(None, configs['train_process']['random_seed'])

    # step 3. data movement: from nfs to local disk
    # dataset_preparation(logger,  configs['data_set'])

    # step 5. init torch.distributed
    init_distributed(configs['train_process']['port'], args.local_rank)

    # step 6. create dataset & dataloader
    class_data = parse_class_info(configs['data_set']['class_name_path'])
    class_num = len(class_data)
    dataloader = gen_dataLoader(logger, 'train', configs, class_data)

    # step 7. network, criteria
    net, criteria_pre, criteria_aux = gen_net_criteria(class_num, configs)
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(newline('network, criteria load.'))

    # step 8. meters, optimizer, lr scheduler
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = \
        gen_meters(configs['train_process']['max_iter'], configs['train_process']['num_aux_heads'])
    optimizer = gen_optimizer(net,
                              configs['train_process']['lr_start'],
                              configs['train_process']['weight_decay'])
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(newline('loss, optimizer load'))
    if has_apex:
        opt_level = 'O1' if configs['train_process']['use_fp16'] else 'O0'
        net, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)

    # DistributedDataParallel training
    net = set_model_dist(net)
    lr_scheduler = WarmupPolyLrScheduler(optimizer, power=0.9,
                                         max_iter=configs['train_process']['max_iter'],
                                         warmup_iter=configs['train_process']['warmup_iters'],
                                         warmup_ratio=0.1, warmup='exp', last_epoch=-1, )

    # step 9. train modules
    train(net, dataloader, optimizer, criteria_pre, criteria_aux, lr_scheduler, time_meter, loss_meter, loss_pre_meter,
          loss_aux_meters, logger, configs)

    # step 10. save the weights
    save_weights(configs['data_set']['model_and_temp_file_save_path'], 'final_model.pth', net, logger)


if __name__ == "__main__":
    main()


