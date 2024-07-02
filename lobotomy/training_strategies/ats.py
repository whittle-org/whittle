from lobotomy.training_strategies.base_strategy import BaseTrainingStrategy


class ATS(BaseTrainingStrategy):
    """
    ATS strategy.

    Updates `random_samples` randomly sampled sub-networks if `self.current_step` is even, otherwise it updates
    the super-network.
    """

    def __init__(self, random_samples: int = 1, **kwargs):
        """
        Initialises an `ATS` strategy.

        Args:
            random_samples: the number of randomly sampled sub-networks to sample and update in each step
            **kwargs: kwargs of `BaseTrainingStrategy`
        """
        super().__init__(**kwargs)
        self.random_samples = random_samples
        self.current_step = 0

    def __call__(self, model, inputs, outputs, **kwargs):
        total_loss = 0
        if self.current_step % 2 == 0:
            # update random sub-networks
            for i in range(self.random_samples):
                config = self.sampler.sample()
                model.select_sub_network(config)
                y_hat = model(inputs)
                loss = self.loss_function(outputs, y_hat)
                loss.backward()
                model.reset_super_network()

                total_loss += loss.item()
        else:
            y_hat = model(inputs)
            loss = self.loss_function(outputs, y_hat)
            loss.backward()
            total_loss = loss.item()
        self.current_step += 1
        return total_loss
