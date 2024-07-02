from lobotomy.training_strategies.base_strategy import BaseTrainingStrategy


class SandwichStrategy(BaseTrainingStrategy):
    """Sandwich strategy.

    In each step, the sandwich strategy updates the super-network, the smallest, and `random_samples` randomly sampled
    sub-networks.

    refs:
        * https://arxiv.org/abs/1903.05134
    """

    def __init__(self, random_samples=2, **kwargs):
        """
        Initialises a `SandwichStrategy`

        Args:
            random_samples: the number of randomly sampled sub-networks to sample and update in each step
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)
        self.random_samples = random_samples

    def __call__(self, model, inputs, outputs, **kwargs):
        total_loss = 0
        # update super-network
        y_hat = model(inputs)
        loss = self.loss_function(outputs, y_hat)
        loss.backward()
        total_loss += loss.item()

        # update random sub-networks
        for i in range(self.random_samples):
            config = self.sampler.sample()
            model.select_sub_network(config)
            y_hat = model(inputs)
            loss = self.loss_function(outputs, y_hat)
            loss.backward()
            model.reset_super_network()
            total_loss += loss.item()

        # smallest network
        config = self.sampler.get_smallest_sub_network()
        model.select_sub_network(config)
        y_hat = model(inputs)
        loss = self.loss_function(outputs, y_hat)
        loss.backward()
        model.reset_super_network()
        total_loss += loss.item()

        return total_loss
