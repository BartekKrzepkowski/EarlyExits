class WandbLogger():
    def __init__(self, wandb, exp_name, epochs):
        self.api_token = "07a2cd842a6d792d578f8e6c0978efeb8dcf7638"
        self.project = f"EarlyExit"
        wandb.login(key=self.api_token)
        wandb.init(
            # Set the project where this run will be logged
            project=self.project,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=exp_name,
            # Track hyperparameters and run metadata
            config={
            "learning_rate": 0.05,
            "architecture": "Resnet18",
            "dataset": "Cifar10",
            "epochs": epochs,
        })
