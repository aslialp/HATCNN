Harmonious Attention Attribute Convolutional Neural Network (HAT-CNN )
===========
HAT-CNN is an modified version of the Harmonious Attention Attribute Convolutional Neural Network (HACNN). It is implemented on top of the existing HA-CNN implementation of torchreid library[1] whose license are given in the repo.


Get started: 30 seconds to Torchreid
-------------------------------------
1. Import ``torchreid``


 ``` 
    import torchreid
 ```
2. Load data manager

 ```    
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='market1501',
        targets='market1501',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop']
    ) 
  ```

3 Build model, optimizer and lr_scheduler

  ```  
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
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
        save_dir='log/resnet50',
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )
```

References
=============
[1] K. Zhou and T. Xiang, “Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch,” arXiv.org, 22-Oct-2019. [Online]. Available: https://arxiv.org/abs/1910.10093. [Accessed: 07-Jan-2021]. 
