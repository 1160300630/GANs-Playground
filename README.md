#### This is a simple GAN Zoo for studying

---

## Mostly using the Comics dataset

### 1. WGAN-GP
- After resizing from 96 to 64 then the images are not belong to 0-255 any more, then doing the "img/127.5 + 1" process not exactly obey the thick transform img to -1~1, don't know weather it matters
- The 'x + eps * (G(z) - x)' step, the dimention of eps is '[batch_size, h, w, c]' or 
'[batchsize, 1, 1, 1] and then broadcast to [batch_size, h, w, c]' ?

### 2. ACGAN in mnist with WGAN-GP loss
#### What is learned
- BN 跟在 RuLU后边是坠吼的
- 网络结构这一块是试出来的，对 G 不加 z 和 label 的 embedding 会崩掉。也可能实际上是每个model对应的lr等超参数没调好。
- 注意loss和网络输出的关系，loss中注意label和logits的位置别弄反了...之前有个bug就是sigmoid算了两遍，结果loss变得有点奇怪...
- WGAN-GP提出来的目的就是为了更容易train GAN，所以WGAN-GP应该比DCGAN等更容易train，超参和模型更容易调整（效果更不更好倒不一定）。可见WGAN-GP更稳定一点。
- 第一种结构不太合理，另外train一个classifier的话没有共享D中卷积后的feature，第二种结构从卷积层的feature后开始分别trainfc组成的classifier可能更合理。


#### TODO
- Add noise to true label. 
- 怎么写出通用的 fc_relu_bn 层...(slim里只有fc_bn_relu)

#### Q
- WGAN-GP的理想loss变化是什么样的？D基本不变G一直降低是否已经optimal了？
- 变为在最后一个fc层分开train class和logits更合理\更难trian...？

#### Result:
![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/AC-GAN(WGAN-GP%20loss).jpg)


### 3. WGAN-GP in mnist

#### Result:
![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/WGAN-GP.jpg)

### 4. WGAN-GP in comic

#### Result:

![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/WGAN-GP(vanilla, 100 epoch).jpg)

###  5. AC-GAN(WGAN-GP loss)

#### Result:

![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/c1.jpg)

![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/c2.jpg)

![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/c3.jpg)

### 