import wandb

import lwclr as lwclr

def train_main(init=True):
    
    args = lwclr.utilities.misc.ret_args()

    dm, trainer, model = lwclr.utilities.misc.environment_loader(args)
    print(args, str(model.backbone.configuration))

    trainer.fit(model, dm)

    #dm.setup('test')
    #trainer.test(test_dataloaders=dm.test_dataloader())

    if init:
        wandb.finish() 

if __name__ == '__main__':
    train_main()
