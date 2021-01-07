from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import CrossEntropyLoss

from ..engine import Engine
import torch
import torch.nn as nn

class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True
    ):
        super(ImageSoftmaxEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs, pids, atts = self.parse_data_for_train(data)
        x,y,z = list(atts.size())
        atts = torch.reshape(atts, (x, y))
        atts = atts.type(torch.FloatTensor)
  
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
            atts = atts.cuda()

        outputs = self.model(imgs)
        loss = self.compute_loss(self.criterion, outputs[0:2], pids)
        #loss2 = self.compute_loss(self.criterion, outputs[1], pids)
        losst = nn.BCEWithLogitsLoss()
        loss2 = losst(outputs[2], atts)
        losses = [loss,loss2]
        total_loss = sum(losses) 
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        #print(atts)
        #print(atts.size())
        #print(outputs[1].size())
        #print(outputs[1])        


        #print(type(atts))
        #print(type(outputs[1]))
        #loss2.backward()
        #self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'loss2': loss2.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_summary
