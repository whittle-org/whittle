import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

from lobotomy.training_strategies.base_strategy import BaseTrainingStrategy


class SandwichStrategy(BaseTrainingStrategy):
    """
    Sandwich strategy.
    """

    def __init__(self, random_samples=2, **kwargs):
        super().__init__(**kwargs)
        self.random_samples = random_samples

    def __call__(self, model, inputs, outputs, **kwargs):

        # update super-network
        y_hat = model(inputs)
        loss = self.loss_function(outputs, y_hat)
        loss.backward()

        # update random sub-networks
        for i in range(self.random_samples):

            config = self.sampler.sample()
            model.select_sub_network(config)
            y_hat = model(inputs)
            loss = self.loss_function(outputs, y_hat)
            loss.backward()
            model.reset_super_network()

        # smallest network
        config = self.sampler.get_smallest_sub_network()
        model.select_sub_network(config)
        y_hat = model(inputs)
        loss = self.loss_function(outputs, y_hat)
        loss.backward()
        model.reset_super_network()

        return loss.item()


def __main__():

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    update_op = SandwichStrategy(loss_function=loss_function)
    model.train()
    overall_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x = batch[:, 0].reshape(-1, 1)
        y = batch[:, 1].reshape(-1, 1)
        x = x.to(device)

        optimizer.zero_grad()

        loss = update_op(model, x, y)

        optimizer.step()

        scheduler.step()
