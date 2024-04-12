# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import torch
import torch.nn as nn
import numpy as np


def register_mask(module, mask):
    hook = lambda _, inputs: (inputs[0] * mask)
    # hook = lambda _, inputs: print(inputs, mask)
    handle = module.register_forward_pre_hook(hook)
    return handle


def loss_function(x, y):
    reproduction_loss = nn.functional.mse_loss(x, y)

    return reproduction_loss


def sample_subnet(hiddem_dim):
    dim = torch.randint(1, hiddem_dim, (1,)).item()
    mask = torch.zeros(hiddem_dim, )
    mask[:dim] = 1
    return mask


def train_epoch(model, train_loader, optimizer, device, **kwargs):
    model.train()
    overall_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x = batch[:, 0].reshape(-1, 1)
        y = batch[:, 1].reshape(-1, 1)
        x = x.to(device)

        optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_function(y, y_hat)

        overall_loss += loss.item()

        loss.backward()

        optimizer.step()

    return overall_loss / (batch_idx + 1)


def train_random_linear(model, train_loader, optimizer, device, total_number_of_steps, current_epoch, random_samples=1):
    model.train()
    overall_loss = 0
    dropout_rate = np.linspace(0.0, 1, total_number_of_steps)
    for batch_idx, batch in enumerate(train_loader):
        x = batch[:, 0].reshape(-1, 1)
        y = batch[:, 1].reshape(-1, 1)
        x = x.to(device)

        optimizer.zero_grad()
        step = current_epoch * (len(train_loader) // batch.shape[0])  + batch_idx
        if np.random.rand() <= dropout_rate[step]:
            # sub-network
            for i in range(random_samples):
                mask = sample_subnet(model.hidden_layer.weight.shape[0])
                handle = register_mask(model.hidden_layer, mask)
                y_hat = model(x)
                loss = loss_function(y, y_hat)
                overall_loss += loss.item()
                loss.backward()
                handle.remove()
        else:
            # super-network
            y_hat = model(x)
            loss = loss_function(y, y_hat)
            overall_loss += loss.item()
            loss.backward()

        optimizer.step()

    return overall_loss / (batch_idx + 1)


def train_ats(model, train_loader, optimizer, device, random_samples=1, **kwargs):
    model.train()
    overall_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x = batch[:, 0].reshape(-1, 1)
        y = batch[:, 1].reshape(-1, 1)
        x = x.to(device)

        optimizer.zero_grad()
        if batch_idx % 2 == 0:
            # sub-network
            for i in range(random_samples):
                mask = sample_subnet(model.hidden_layer.weight.shape[0])
                handle = register_mask(model.hidden_layer, mask)
                y_hat = model(x)
                loss = loss_function(y, y_hat)
                overall_loss += loss.item()
                loss.backward()
                handle.remove()
        else:
            # super-network
            y_hat = model(x)
            loss = loss_function(y, y_hat)
            overall_loss += loss.item()
            loss.backward()

        optimizer.step()

    return overall_loss / (batch_idx + 1)

def train_random(model, train_loader, optimizer, device, random_samples=1, **kwargs):
    model.train()
    overall_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x = batch[:, 0].reshape(-1, 1)
        y = batch[:, 1].reshape(-1, 1)
        x = x.to(device)

        optimizer.zero_grad()
        # sub-network
        for i in range(random_samples):
            mask = sample_subnet(model.hidden_layer.weight.shape[0])
            handle = register_mask(model.hidden_layer, mask)
            y_hat = model(x)
            loss = loss_function(y, y_hat)
            overall_loss += loss.item()
            loss.backward()
            handle.remove()

        optimizer.step()

    return overall_loss / (batch_idx + 1)


def train_sandwich(model, train_loader, optimizer, device, random_samples=2, **kwargs):
    model.train()
    overall_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x = batch[:, 0].reshape(-1, 1)
        y = batch[:, 1].reshape(-1, 1)
        x = x.to(device)

        optimizer.zero_grad()

        # super-network
        y_hat = model(x)
        loss = loss_function(y, y_hat)
        overall_loss += loss.item()
        loss.backward()

        # sub-network
        for i in range(random_samples):
            mask = sample_subnet(model.hidden_layer.weight.shape[0])
            handle = register_mask(model.hidden_layer, mask)
            y_hat = model(x)
            loss = loss_function(y, y_hat)
            overall_loss += loss.item()
            loss.backward()
            handle.remove()

        # smallest network
        mask = torch.zeros(model.hidden_layer.weight.shape[0], )
        mask[0] = 1
        handle = register_mask(model.hidden_layer, mask)
        y_hat = model(x)
        loss = loss_function(y, y_hat)
        overall_loss += loss.item()
        loss.backward()
        handle.remove()

        optimizer.step()

    return overall_loss / (batch_idx + 1)


def train_sandwich_kd(model, train_loader, optimizer, device, random_samples=2, **kwargs):
    model.train()
    overall_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x = batch[:, 0].reshape(-1, 1)
        y = batch[:, 1].reshape(-1, 1)
        x = x.to(device)

        optimizer.zero_grad()

        # super-network
        y_teacher = model(x)
        loss = loss_function(y, y_teacher)
        overall_loss += loss.item()
        loss.backward()

        y_teacher = y_teacher.detach()

        # sub-network
        for i in range(random_samples):
            mask = sample_subnet(model.hidden_layer.weight.shape[0])
            handle = register_mask(model.hidden_layer, mask)
            y_hat = model(x)
            loss = loss_function(y, y_hat) + loss_function(y_teacher, y_hat)
            overall_loss += loss.item()
            loss.backward()
            handle.remove()

        # smallest network
        mask = torch.zeros(model.hidden_layer.weight.shape[0], )
        mask[0] = 1
        handle = register_mask(model.hidden_layer, mask)
        y_hat = model(x)
        loss = loss_function(y, y_hat) + loss_function(y_teacher, y_hat)
        overall_loss += loss.item()
        loss.backward()
        handle.remove()

        optimizer.step()

    return overall_loss / (batch_idx + 1)
