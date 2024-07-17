from lobotomy.training_strategies.base_strategy import (
    BaseTrainingStrategy,
)


class RandomStrategy(BaseTrainingStrategy):
    """
    Random strategy.
    """

    def __init__(self, random_samples=1, **kwargs):
        super().__init__(**kwargs)
        self.random_samples = random_samples

    def __call__(self, model, inputs, outputs, **kwargs):
        total_loss = 0
        y_supernet = model(inputs).detach()
        # update random sub-networks
        for i in range(self.random_samples):
            config = self.sampler.sample()
            model.select_sub_network(config)
            y_hat = model(inputs)
            if self.kd_loss is not None:
                loss = self.kd_loss(y_hat, outputs, y_supernet)
            else:
                loss = self.loss_function(y_hat, outputs)
            loss.backward()
            model.reset_super_network()

            total_loss += loss.item()
        return total_loss
