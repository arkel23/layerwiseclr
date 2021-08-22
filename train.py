import argparse, os

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import lwclr as lwclr

def load_trainer(args):
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', 
    dirpath=args.results_dir, filename=args.run_name, save_top_k=1, 
    mode='max', save_weights_only=True, every_n_epochs=args.save_checkpoint_freq)
    
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.logger = wandb_logger
    
    if trainer.max_steps:
        total_steps = trainer.max_steps
    else:
        total_steps = trainer.max_epochs * len(dm.train_dataloader())
    args.total_steps = total_steps

    return trainer


def environment_loader(args, init=True):
    
    # set up W&B logger
    if init:
        os.makedirs(args.results_dir, exist_ok=True)
        wandb.init(config=args)
        wandb.run.name = args.run_name
        wandb_logger = WandbLogger(name=args.run_name)
        
    # seed everything
    pl.seed_everything(seed=args.seed)

    # prepare datamodule
    dm = lwclr.utilities.datamodules.return_prepared_dm(args)

    # setup model and trainer
    trainer = load_trainer(args, dm)
    model = lwclr.models.LitClassifier(args, total_steps, **vars(args))

    return dm, trainer, model


def train_main(init=True):
    
    args = lwclr.utilities.misc.ret_args()
    #dm, trainer, model = environment_loader(args)
    
    # train, validate
    #trainer.fit(model, dm)
    #trainer.test(test_dataloaders=dm.test_dataloader())

    #if init:
    #    wandb.finish() 

if __name__ == '__main__':
    train_main()
