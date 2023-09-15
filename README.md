Harmonious Attention Attribute Convolutional Neural Network (HAT-CNN )
===========
HAT-CNN is a modified version of the Harmonious Attention Attribute Convolutional Neural Network (HACNN). It is implemented on top of the existing HA-CNN implementation of torchreid library[1] whose license are given in the repo.


1. Clone ``torchreid`` repo.
 ``` 
!git clone https://github.com/KaiyangZhou/deep-person-reid.git 
 ``` 

2. Modify the following code files with the corresponding file codes in this repo.
 ``` 
 deep-person-reid/torchreid/data/datasets/image/market1501.py
 deep-person-reid/torchreid/data/datasets/dataset.py
 deep-person-reid/torchreid/engine/engine.py
 deep-person-reid/torchreid/engine/image/softmax.py
 deep-person-reid/torchreid/models/hacnn.py
 ``` 
 
3. Upload the ``annots.csv`` to the workspace.
 

3. Import ``torchreid``

 ``` 
    import torchreid
 ```
4. Load data manager

 ```    
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='market1501',
        targets='market1501',
        height=160,
        width=64,
        batch_size_train=16,
        batch_size_test=16,
        transforms=None
    ) 
  ```

3 Build model, optimizer and lr_scheduler

  ```  
    model = torchreid.models.build_model(
        name='hacnn',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='sgd',
        lr=0.03
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='multi_step',
        stepsize=[150, 225]
    )
   ```
4. Build engine

```
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )
```
5. Run training and test

```
    engine.run(
        save_dir='log/hacnn',
        max_epoch=240,
        eval_freq=20,
        print_freq=404,
        test_only=False
    )
```

References
=============
[1] K. Zhou and T. Xiang, “Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch,” arXiv.org, 22-Oct-2019. [Online]. Available: https://arxiv.org/abs/1910.10093. [Accessed: 07-Jan-2021]. 
