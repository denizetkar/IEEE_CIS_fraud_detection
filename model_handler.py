import gc

import torch
import torch.nn as nn


class FixedInputFixedOutputModelHandler:

    def __init__(self, model_module, loss_module, optim_class,
                 model_args=(), loss_args=(), optim_args=(),
                 device=torch.device('cpu'), l1_regularization_weight=0.01):
        if isinstance(model_args, dict):
            self.model = model_module(**model_args).to(device)
        elif hasattr(model_args, '__iter__'):
            self.model = model_module(*model_args).to(device)
        else:
            self.model = model_module().to(device)

        if isinstance(loss_args, dict):
            self.loss_func = loss_module(**loss_args)
        elif hasattr(loss_args, '__iter__'):
            self.loss_func = loss_module(*loss_args)
        else:
            self.loss_func = loss_module()

        if isinstance(optim_args, dict):
            self.optimizer = optim_class(self.model.parameters(), **optim_args)
        elif hasattr(optim_args, '__iter__'):
            self.optimizer = optim_class(self.model.parameters(), *optim_args)
        else:
            self.optimizer = optim_class(self.model.parameters())

        self.device = device
        self.l1_regularization_weight = l1_regularization_weight
        self.l1_loss_func = nn.L1Loss()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
        except FileNotFoundError:
            pass

    def batch_to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        return [self.batch_to_device(batch_component) for batch_component in batch]

    def train(self, in_tgt_generator, eval_in_tgt_generator=None, update_per_eval=50,
              step_per_update=1, update_per_save=10, save_path='',
              update_per_verbose=50, verbose=True, summary_func_list=None):
        if summary_func_list is None:
            summary_func_list = []

        step_count, optim_count, update_loss = 0, 0, 0.0
        self.model.train()
        for step_count, (input_batch, target_batch) in enumerate(in_tgt_generator, start=1):
            input_batch, target_batch = self.batch_to_device(input_batch), self.batch_to_device(target_batch)
            output_batch = self.model(input_batch)
            loss = self.loss_func(output_batch, target_batch) / step_per_update
            loss.backward()
            update_loss += loss.item()
            if verbose:
                # Let user defined summary functions run
                for summary_func in summary_func_list:
                    summary = summary_func(model=self.model, input_batch=input_batch, output_batch=output_batch,
                                           loss=loss.item(), step_count=step_count, step_per_update=step_per_update,
                                           update_per_verbose=update_per_verbose)
                    if summary is not None:
                        summary_name, summary_val = summary
                        print('At step: ' + str(step_count) + ' ' + summary_name + ': ' + str(summary_val))
            # If it is time to update, then update the model
            if step_count % step_per_update == 0:
                # Calculate L1 norm of the model parameters
                l1_loss = self.l1_regularization_weight * sum(
                    self.l1_loss_func(param, torch.zeros_like(param)) for param in self.model.parameters())
                l1_loss.backward()
                # Take an optimization step
                self.optimizer.step()
                self.optimizer.zero_grad()
                optim_count += 1
                if optim_count % update_per_verbose == 0 and verbose:
                    print('At step: ' + str(step_count) + ' Loss: ' + str(update_loss))
                update_loss = 0.0
                if optim_count % update_per_eval == 0 and eval_in_tgt_generator is not None and verbose:
                    eval_loss = self.eval(in_tgt_generator=eval_in_tgt_generator)
                    self.model.train()
                    print('Eval loss @ step: ' + str(step_count) + ' is ' + str(eval_loss))
                if optim_count % update_per_save == 0 and save_path:
                    self.save(save_path)

                # Open up space for new variables in the next iteration
                del l1_loss
                gc.collect()

            # Open up space for new variables in the next iteration
            del loss, output_batch, input_batch, target_batch
            gc.collect()

        # Check for unaccounted loss gradients
        unaccounted_steps = step_count % step_per_update
        if unaccounted_steps > 0:
            # Adjust the scaling of gradients to take average gradient
            # over 'unaccounted_steps' number of steps
            for param in self.model.parameters():
                param.grad *= step_per_update / unaccounted_steps
            # Calculate L1 norm of the model parameters
            l1_loss = self.l1_regularization_weight * torch.stack(
                [self.l1_loss_func(param, torch.zeros_like(param)) for param in self.model.parameters()]).mean()
            l1_loss.backward()
            # Take an optimization step
            self.optimizer.step()
            self.optimizer.zero_grad()
            if save_path:
                self.save(save_path)

    def eval(self, in_tgt_generator):
        self.model.eval()
        mean_loss = 0.0
        for n, (input_batch, target_batch) in enumerate(in_tgt_generator, start=1):
            input_batch, target_batch = self.batch_to_device(input_batch), self.batch_to_device(target_batch)
            output_batch = self.model(input_batch)
            loss = self.loss_func(output_batch, target_batch)
            mean_loss = mean_loss + (loss.item() - mean_loss) / n
            # Open up space for new variables in the next iteration
            del loss, output_batch, input_batch, target_batch
            gc.collect()
        return mean_loss

    def predict(self, in_id_generator):
        self.model.eval()
        for input_batch, id_batch in in_id_generator:
            input_batch = self.batch_to_device(input_batch)
            yield self.model(input_batch).detach(), id_batch.detach()

    def accuracy(self, in_tgt_generator):
        self.model.eval()
        num_of_correct_pred, num_of_samples = 0, 0
        for input_batch, target_batch in in_tgt_generator:
            input_batch, target_batch = self.batch_to_device(input_batch), self.batch_to_device(target_batch)
            output_batch = self.model(input_batch)
            num_of_correct_pred += torch.sum(torch.argmax(output_batch.detach(), dim=-1) == target_batch).item()
            num_of_samples += target_batch.shape[0]
        return num_of_correct_pred / num_of_samples
