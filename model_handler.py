import torch


class FixedInputFixedOutputModelHandler:

    def __init__(self, model_module, loss_module, optim_class,
                 model_args=(), loss_args=(), optim_args=(),
                 device=torch.device('cpu')):
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

    def train(self, in_tgt_generator, update_per_step=1, save_per_update=10, save_path='',
              verbose_per_update=50, verbose=True, summary_func_list=None):
        if summary_func_list is None:
            summary_func_list = []
        optim_count, update_loss = 0, 0.0
        self.model.train()
        for step_count, (input_batch, target_batch) in enumerate(in_tgt_generator, start=1):
            input_batch, target_batch = self.batch_to_device(input_batch), self.batch_to_device(target_batch)
            output_batch = self.model(input_batch)
            loss = self.loss_func(output_batch, target_batch) / update_per_step
            loss.backward()
            update_loss += loss.item()
            if verbose:
                # Let user defined summary functions run
                for summary_func in summary_func_list:
                    summary = summary_func(model=self.model, input_batch=input_batch, output_batch=output_batch,
                                           loss=loss.item(), step_count=step_count, update_per_step=update_per_step,
                                           verbose_per_update=verbose_per_update)
                    if summary is not None:
                        print('At step: ' + str(step_count) + ' ' + summary[0] + ': ' + str(summary[1]))
            # If it is time to update, then update the model
            if step_count % update_per_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                optim_count += 1
                if optim_count % verbose_per_update == 0 and verbose:
                    print('At step: ' + str(step_count) + ' Loss: ' + str(update_loss))
                update_loss = 0.0
                if optim_count % save_per_update == 0 and save_path:
                    self.save(save_path)

    def eval(self, in_tgt_generator):
        self.model.eval()
        mean_loss = 0.0
        for n, (input_batch, target_batch) in enumerate(in_tgt_generator, start=1):
            input_batch, target_batch = self.batch_to_device(input_batch), self.batch_to_device(target_batch)
            output_batch = self.model(input_batch)
            loss = self.loss_func(output_batch, target_batch)
            mean_loss = mean_loss + (loss.item() - mean_loss) / n
        return mean_loss

    def predict(self, in_generator):
        self.model.eval()
        for input_batch in in_generator:
            yield self.model(input_batch).detach().cpu().numpy()
