from lobotomy.training_strategies.base_strategy import BaseTrainingStrategy


class RandomStrategy(BaseTrainingStrategy):
    """
    Random strategy.

    Randomly samples and updates `random_samples` sub-networks in each step.
    """

    def __init__(self, random_samples=1, **kwargs):
        """
        Initialises a `RandomStrategy`

        Args:
            random_samples: the number of randomly sampled sub-networks to sample and update in each step
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)
        self.random_samples = random_samples

    def __call__(self, model, inputs, outputs, **kwargs):
        """Updates randomly sampled sub-networks in each step."""
        total_loss = 0
        for i in range(self.random_samples):
            config = self.sampler.sample()
            model.select_sub_network(config)
            y_hat = model(inputs)
            loss = self.loss_function(outputs, y_hat)
            loss.backward()
            model.reset_super_network()

            total_loss += loss.item()
        return total_loss
