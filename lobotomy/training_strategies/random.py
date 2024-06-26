from lobotomy.training_strategies.base_strategy import BaseTrainingStrategy


class RandomStrategy(BaseTrainingStrategy):
    """
    Random strategy.
    """

    def __init__(self, random_samples=1, **kwargs):
        super().__init__(**kwargs)
        self.random_samples = random_samples

    def __call__(self, model, inputs, outputs, **kwargs):
        total_loss = 0
        # update random sub-networks
        for i in range(self.random_samples):
            config = self.sampler.sample()
            model.select_sub_network(config)
            y_hat = model(inputs)
            loss = self.loss_function(outputs, y_hat)
            loss.backward()
            model.reset_super_network()

            total_loss += loss.item()
        return total_loss
