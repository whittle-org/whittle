from lobotomy.training_strategies.base_strategy import BaseTrainingStrategy


class StandardStrategy(BaseTrainingStrategy):
    """
    Standard strategy.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, model, inputs, outputs, **kwargs):
        y_hat = model(inputs)
        loss = self.loss_function(outputs, y_hat)
        loss.backward()
        return loss.item()
