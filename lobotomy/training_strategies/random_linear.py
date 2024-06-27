import numpy as np

from lobotomy.training_strategies.base_strategy import BaseTrainingStrategy


class RandomLinearStrategy(BaseTrainingStrategy):
    """
    Random linear strategy.
    """

    def __init__(self, total_number_of_steps: int, random_samples: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.random_samples = random_samples
        self.total_number_of_steps = total_number_of_steps
        self.current_step = 0
        self.rate = np.linspace(0.0, 1, total_number_of_steps)

    def __call__(self, model, inputs, outputs, **kwargs):
        total_loss = 0
        if np.random.rand() <= self.rate[self.current_step]:
            # update random sub-networks
            for i in range(self.random_samples):
                config = self.sampler.sample()
                model.select_sub_network(config)
                y_hat = model(inputs)
                loss = self.loss_function(y_hat, outputs)
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
