import os
import wandb

import lwclr as lwclr

def train_main(init=True):
    
    args = lwclr.utilities.loaders.ret_args()

    dm, trainer, model = lwclr.utilities.loaders.environment_loader(args)
    print(args, str(model.backbone.configuration))

    trainer.fit(model, dm)

    dm.setup('test')
    #trainer.test(datamodule=dm, ckpt_path=os.path.join(args.results_dir, args.run_name, 'last.ckpt'))
    trainer.test(datamodule=dm)

    if init:
        wandb.finish() 

if __name__ == '__main__':
    train_main()
