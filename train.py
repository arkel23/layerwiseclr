import os
import wandb

import lwclr as lwclr

def train_main(init=True):
    
    args = lwclr.utilities.misc.ret_args()

    dm, trainer, model = lwclr.utilities.misc.environment_loader(args)
    print(args, str(model.backbone.configuration))

    trainer.fit(model, dm)

    dm.setup('test')
    trainer.test(datamodule=dm, ckpt_path=os.path.join(args.results_dir, args.run_name, 'last.ckpt'))

    if init:
        wandb.finish() 

if __name__ == '__main__':
    train_main()
