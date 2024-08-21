from __future__ import annotations

from whittle.training_strategies.base_strategy import BaseTrainingStrategy


class ATS(BaseTrainingStrategy):
    """
    ATS strategy.

    Follows the approach by Mohtashami et al. and updates a set of randomly sampled sub-networks if
    if the current step is even, otherwise it updates the super-network.

    refs:
        Masked Training of Neural Networks with Partial Gradients
        Amirkeivan Mohtashami, Martin Jaggi, Sebastian Stich
        Proceedings of The 25th International Conference on Artificial Intelligence and Statistics
        https://arxiv.org/abs/2106.08895
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
        """
        Updates a set of randomly sampled sub-networks if the current step is odd. Else, it updates the
        super-network.
        """
        total_loss = 0
        y_supernet = model(inputs)
        if self.current_step % 2 == 0:
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
        else:
            y_hat = model(inputs)
            if self.kd_loss is not None:
                loss = self.kd_loss(y_hat, outputs, y_supernet)
            else:
                loss = self.loss_function(y_hat, outputs)
            loss.backward()
            total_loss = loss.item()
        self.current_step += 1
        return total_loss
