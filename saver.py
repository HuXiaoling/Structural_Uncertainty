import os
import shutil
import torch
from collections import OrderedDict
import glob
import json

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('experiments', args.dataset)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = len([lists for lists in os.listdir(self.directory) if lists.startswith('experiment')]) if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # Save args
        with open(os.path.join(self.experiment_dir, 'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        shutil.copyfile(filename, os.path.join(self.directory, 'model_last.pth.tar'))
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_loss = [10000]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        try:
                            with open(path, 'r') as f:
                                loss = float(f.readline())
                                previous_loss.append(loss)
                        except:
                            pass
                    else:
                        continue
                min_loss = min(previous_loss)
                if best_pred < min_loss:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['dataset'] = self.args.dataset
        p['lr'] = self.args.lr
        p['epochs'] = self.args.epochs

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()