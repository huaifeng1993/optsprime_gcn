import os, os.path as osp
from copy import copy
import torch, time, argparse
import torch.distributed as dist
from  mmcv import Config,DictAction
from optsprime.utils import get_root_logger
from optsprime.models import build_framework
from optsprime.datasets import build_dataset,build_dataloader
from optsprime.optimizers import build_optimizer,build_lrscheduler
#from optsprime.utils import OPENOCC_LOSS
from optsprime.utils import create_scheduler,AverageMeter,MeanIoU,revise_ckpt,revise_ckpt_2,dtypeLut
import warnings
import mmcv

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    cfg.distributed = args.dist

    # init DDP
    if cfg.distributed:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20506")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank
        )
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        # disable print from none-0 processes
        if dist.get_rank() != 0:
            import builtins
            builtins.print = pass_print
    else:
        rank = 0
        cfg.gpu_ids = [0]
    
    # dump configuration
    if local_rank == 0 and rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    # configure logging
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    # build modelq
    my_model = build_framework(cfg.model)

    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if torch.cuda.is_available():
        logger.info('cuda is available')
        my_model=my_model.cuda()
    else:
        logger.info('cuda is not available. the model will be trained on cpu')

    if cfg.distributed and torch.cuda.is_available():
        # If the configuration specifies that the model should be distributed, then we use the DistributedDataParallel module from PyTorch to parallelize the model across multiple GPUs.
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        logger.info('done ddp model')

    # generate datasets
    train_dataset=build_dataset(cfg.data.train)
    val_dataset=build_dataset(cfg.data.val)

    train_dataset_loader = build_dataloader(train_dataset,cfg=cfg.data)
    val_dataset_loader=build_dataloader(val_dataset,cfg=cfg.data)
    # get metric calculator
    # label_str = train_dataset_loader.dataset.loader.nuScenes_label_name
    # metric_label = cfg.unique_label
    # metric_str = [label_str[x] for x in metric_label]
    # metric_ignore_label = cfg.metric_ignore_label
    # CalMeanIou_pts = MeanIoU(metric_label, metric_ignore_label, metric_str, 'pts')

    # get optimizer, loss, scheduler
    optimizer = build_optimizer(my_model, cfg.optimizer)
    #multi_loss_func = OPENOCC_LOSS.build(cfg.loss)
    scheduler = build_lrscheduler(optimizer,cfg.lr_schedule)
    # resume and load
    epoch = 0
    best_val_miou_pts = 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        logger.info(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_miou_pts' in ckpt:
            best_val_miou_pts = ckpt['best_val_miou_pts']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            print('removing img_neck.lateral_convs and img_neck.fpn_convs')
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
        

    # training
    print_freq = cfg.train_cfg.print_freq
    max_num_epochs = cfg.train_cfg.max_epochs
    lossMeter = AverageMeter()
    while epoch < max_num_epochs:
        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        lossMeter.reset()
        data_time_s = time.time()
        time_s = time.time()

        for i_iter, inputs in enumerate(train_dataset_loader):
            extra_inputs={}
            for key, value in inputs.items():
                if key in cfg.data.inputs_extkeys:
                    extra_inputs[key] = value
            new_inputs=dict(inputs=inputs,
                            kwargs=extra_inputs)
            data_time_e = time.time()
            outputs=my_model.train_step(new_inputs,optimizer)
            loss=outputs["loss"]
            log_vars=outputs["log_vars"]
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.train_cfg.clip_norm)
            optimizer.step()
            scheduler.step()
            time_e = time.time()
            global_iter += 1
            if i_iter % print_freq == 0 and local_rank == 0 and rank == 0:
                log_info=""
                for key, value in log_vars.items():
                    log_info+="{}: {:.3f} ".format(key,value)
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter:%5d/%d: %s, grad_norm: %.1f, lr: %.7f, time: %.3f(%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader), log_info,grad_norm,lr,
                    time_e - time_s, data_time_e - data_time_s))
            data_time_s = time.time()
            time_s = time.time()
        
        # save checkpoint
        if local_rank == 0 and rank == 0:
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
                'best_val_miou_pts': best_val_miou_pts,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            mmcv.symlink(save_file_name, dst_file)
        
        # eval
        if epoch%cfg.train_cfg.val_step==0:
            my_model.eval()
            with torch.no_grad():
                for i_iter_val, inputs in enumerate(val_dataset_loader):
                    extra_inputs={}
                    for key, value in inputs.items():
                        if key in cfg.data.inputs_extkeys:
                            extra_inputs[key] = value
                    new_inputs=dict(inputs=inputs,
                                    kwargs=extra_inputs)
                    data_time_e = time.time()
                    outputs = my_model.val_step(new_inputs)
                    log_info=""
                    for key, value in outputs["log_vars"].items():
                        log_info+="{}: {:.3f} ".format(key,value)

                    if i_iter_val % print_freq == 0 and local_rank == 0 and rank == 0:
                        logger.info('[EVAL] Epoch %d Iter %5d: %s'%(
                            epoch, i_iter_val,log_info))
                    
        # val_miou_pts = CalMeanIou_pts._after_epoch()
        # if best_val_miou_pts < val_miou_pts:
        #     best_val_miou_pts = val_miou_pts
        # logger.info('Current val miou pts is %.3f while the best val miou pts is %.3f' %
        #         (val_miou_pts, best_val_miou_pts))
        # logger.info('Current val loss is %.3f' %
        #         (lossMeter.avg))
        
        epoch += 1
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/config_gcn.py')
    parser.add_argument('--work-dir', type=str, default='./out/test')
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--resume-from', type=str, default='')

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if args.dist:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)