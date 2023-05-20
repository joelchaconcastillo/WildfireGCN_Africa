import torch
import math
import os
import time
import copy
import numpy as np
import scipy.sparse as sp
from logger import get_logger
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn import metrics

#from metrics import All_Metrics
def Binary_metrics(targets, probs, threshold=0.5):
        preds = probs > threshold
        tn, fp, fn, tp = metrics.confusion_matrix(targets, preds).ravel()
        auc = roc_auc_score(targets, probs)
        aucpr = average_precision_score(targets, probs)
        summary = classification_report(targets, preds, digits=3, output_dict=True)['1.0']
        summary['AUC'] = auc
        summary['AUCPR'] = aucpr
        summary['TP'] = tp
        summary['FP'] = fp
        summary['TN'] = tn
        summary['FN'] = fn
        summary['accuracy'] = (tp+tn)/float(tp+fp+tn+fn)
        return summary
class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader1, test_loader2,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader1 = test_loader1
        self.test_loader2 = test_loader2
        self.scaler = scaler
        self.args = args
        self.device = args.device
        self.lr_scheduler = lr_scheduler
        self.number_minibatches = int(args.batch_size/args.minbatch_size)
        self.minbatch_size = args.minbatch_size
        self.batch_size = args.batch_size
        self.start_epoch = args.start_epoch
        self.train_per_epoch = len(train_loader)
        self.best_model = args.best_model
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
        #log
        if os.path.isdir(args.log_dir) == False and not args.debug:
           os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
        self.logger.info('log dir: {}'.format(args.log_dir))
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        #if not args.debug:
        self.logger.info("Argument: %r", args)
        for arg, value in sorted(vars(args).items()):
            self.logger.info("Argument %s: %r", arg, value)



    def predictions_model(self, model, data_loader):
        self.model.eval()
        targets = np.array([])
        probs = np.array([])
        with torch.no_grad():
            for batch_idx, (data, label, ZPI) in enumerate(data_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                ZPI = ZPI.to(self.device)
                output = self.model(data, ZPI)
                targets= np.concatenate((targets, label.cpu().numpy()))
                probs = np.concatenate((probs, (torch.exp(output)[:, 1]).cpu().numpy()))                
        return targets, probs

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        countBatches = 0.
        preds = np.array([])
        targets = np.array([])
        probs = np.array([])
        
        with torch.no_grad():
            for batch_idx, (data, label, ZPI) in enumerate(val_dataloader):
                data = data.to(self.device)
                label = label.to(self.device)
                ZPI = ZPI.to(self.device)
                output = self.model(data, ZPI)
                preds = np.concatenate((preds, torch.argmax(output, dim=1).cpu().numpy()))
                targets= np.concatenate((targets, label.cpu().numpy()))
                probs = np.concatenate((probs, (torch.exp(output)[:, 1]).cpu().numpy()))

                loss = self.loss(output, label)

                loss = loss/self.batch_size
                
                if ( (batch_idx+1)%self.number_minibatches == 0 ) or ((batch_idx+1) == len(val_dataloader)):
                    countBatches +=1.
                if not torch.isnan(loss):
                   total_val_loss += loss.item()
        summary = Binary_metrics(targets, probs, 0.5)
#        tn, fp, fn, tp = metrics.confusion_matrix(targets, preds).ravel()
#        auc = roc_auc_score(targets, probs)
#        aucpr = average_precision_score(targets, probs)
#        summary = classification_report(targets, preds, digits=3, output_dict=True)['1.0']
#        summary['AUC']=auc
#        summary['AUCPR']=aucpr
#        summary['TP']=tp
#        summary['FP']=fp
##        summary['TN']=tn
#        summary['FN']=fn
        self.logger.info("\n metrics validation: {} \n".format(summary))

        val_loss = total_val_loss / countBatches
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        self.logger.info("acc: {}\n".format(summary['accuracy']))
#        sensitivity = tp/float(tp+fn) 
#        specificity = tp/float(tn+fp)
#        balancedAcc = (sensitivity+specificity)/2.
#        self.logger.info("balanced acc: {}\n".format(balancedAcc))
        return summary['accuracy'], val_loss
 #       return val_loss, -balancedAcc
#        return val_loss, -summary['f1-score']

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_loss_batch = 0
        countBatches = 0
        self.optimizer.zero_grad()
        for batch_idx, (data, label, ZPI) in enumerate(self.train_loader):
            data = data.to(self.device)
            label = label.to(self.device)
            ZPI = ZPI.to(self.device)
            output = self.model(data, ZPI)
            loss = self.loss(output, label)
#            l1_lambda = 0.01
#            l1_norm = sum(torch.abs(p).sum()
#                  for p in self.model.parameters())
#            loss = loss + l1_lambda*l1_norm
            loss = loss/self.batch_size
            total_loss_batch += loss.item()
           # self.logger.info('{}...'.format(loss.item()))
            loss.backward()
            if ( (batch_idx+1)%self.number_minibatches == 0 ) or ((batch_idx+1) == len(self.train_loader)):
               # add max grad clipping
               if self.args.grad_norm:
                  torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
               self.optimizer.step()
               self.optimizer.zero_grad()
               total_loss += total_loss_batch

               #log information
#               if (int((batch_idx+1)/self.number_minibatches)) % self.args.log_step == 0:
               self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(epoch, batch_idx, self.train_per_epoch, total_loss_batch))
               total_loss_batch = 0
               countBatches +=1.
        train_epoch_loss = total_loss/countBatches
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f} '.format(epoch, train_epoch_loss))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        if self.best_model == None:
          best_loss = float('inf')
        else:
          current = copy.deepcopy(self.model.state_dict())
          self.model.load_state_dict(self.best_model)
          _, best_loss = self.val_epoch(self.start_epoch, self.val_loader) # 
          self.model.load_state_dict(current)
        val_epoch_loss = best_loss
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(self.start_epoch, self.args.epochs + 1): # 
            self.current_epoch = epoch
            epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            self.logger.info('\nEpoch time elapsed: {}\n'.format(time.time() - epoch_time))
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            _, val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e7:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
                self.train_loader.sampler.shuffle()
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                self.logger.info('*********************************Current best model saved! {}'.format(self.best_path))
                self.best_model = copy.deepcopy(self.model.state_dict())
            self.save_checkpoint()
                #torch.save(best_model, self.best_path)

            # apply the best model to test dataset
            # test
#            self.model.load_state_dict(best_model)

#            if best_state == True:
            if epoch % 5 == 0 :
               current = copy.deepcopy(self.model.state_dict())
               self.model.load_state_dict(self.best_model)
               # self.val_epoch(self.args.epochs, self.test_loader)
               self.test(self.model, self.args, self.test_loader1, self.scaler, self.logger)
               self.test(self.model, self.args, self.test_loader2, self.scaler, self.logger)
               self.model.load_state_dict(current)


        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
#        if not self.args.debug:
#            torch.save(best_model, self.best_path)
#            self.logger.info("Saving current best model to " + self.best_path)

        #test
        self.model.load_state_dict(self.best_model)
        ####check several thresholds...
#        thresholds = np.concatenate((np.arange(0.2, 1.0, 0.05),np.arange(0.45,0.55, 0.01)))
#        thresholds = np.arange(0.2, 1.0, 0.05)
#        fscores = np.zeros(shape=len(thresholds))
        ## fit the model..
#        for index, element in enumerate(thresholds):
#          targets, probs, = self.predictions_model(self.model, self.val_loader)
#          fscores[index] = Binary_metrics(targets, probs, element)['f1-score']
#          self.logger.info("threshold... {} fscore.. {}".format(element, fscores[index]))
#          self.test(self.model, self.args, self.test_loader1, self.scaler, self.logger, element)
#          self.test(self.model, self.args, self.test_loader2, self.scaler, self.logger, element)

#        maxthreshold = thresholds[np.argmax(fscores)]
#        self.logger.info("Best threshold... {}".format(maxthreshold))
#        self.test(self.model, self.args, self.test_loader1, self.scaler, self.logger, maxthreshold)
#        self.test(self.model, self.args, self.test_loader2, self.scaler, self.logger, maxthreshold)
        self.logger.info("Threshold of 0.5")
        self.test(self.model, self.args, self.test_loader1, self.scaler, self.logger, 0.5)
        self.test(self.model, self.args, self.test_loader2, self.scaler, self.logger, 0.5)


    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'state_dict_best': self.best_model, 
            'optimizer': self.optimizer.state_dict(),
            'config': self.args,
            'loss' : self.loss,
            'lr_scheduler' : self.lr_scheduler,
            'epoch': self.current_epoch+1, ###it is saved at the end..
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, logger, threshold=0.5, path=None):
#        if path != None:
#            check_point = torch.load(path)
##            state_dict = check_point['state_dict']
#            args = check_point['config']
#            model.load_state_dict(state_dict)
#            model.to(args.device)
        model.eval()
        preds = np.array([])
        targets = np.array([])
        probs = np.array([])
        with torch.no_grad():
            for batch_idx, (data, label, ZPI) in enumerate(data_loader):
                data = data.to(args.device)
                label = label.to(args.device)
                ZPI = ZPI.to(args.device)
                output = model(data, ZPI)
                preds = np.concatenate((preds, torch.argmax(output, dim=1).cpu().numpy()))
                targets= np.concatenate((targets, label.cpu().numpy()))
                probs = np.concatenate((probs, (torch.exp(output)[:, 1]).cpu().numpy()))
#        np.save('./{}_true.npy'.format(args.dataset), targets)
#        np.save('./{}_pred.npy'.format(args.dataset), preds)
        summary = Binary_metrics(targets, probs, threshold)

        logger.info("\n Testing metrics {} \n".format(summary))
        logger.info("acc: {}\n".format(summary['accuracy']))
        return summary
