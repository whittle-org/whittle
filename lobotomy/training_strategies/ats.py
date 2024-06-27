from lobotomy.training_strategies.base_strategy import BaseTrainingStrategy


class ATS(BaseTrainingStrategy):
    """
    ATS strategy.
    """

    def __init__(self, random_samples: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.random_samples = random_samples
        self.current_step = 0

    def __call__(self, model, inputs, outputs, **kwargs):
        total_loss = 0
        if self.current_step % 2 == 0:
            # update random sub-networks
            for i in range(self.random_samples):
                config = self.sampler.sample()
                print(config)
                model.select_sub_network(config)
                y_hat = model(inputs)
                loss = self.loss_function(y_hat, outputs)
                loss.backward()
                model.reset_super_network()

                total_loss += loss.item()
        else:
            y_hat = model(inputs)
            loss = self.loss_function(y_hat, outputs)
            loss.backward()
            total_loss = loss.item()
        self.current_step += 1
        return total_loss
