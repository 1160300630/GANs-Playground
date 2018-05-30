#### This is a simple GAN Zoo for studying

---

## Mostly using the Comics dataset

### 1. WGAN-GP
- After resizing from 96 to 64 then the images are not belong to 0-255 any more, then doing the "img/127.5 + 1" process not exactly obey the thick transform img to -1~1, don't know weather it matters
- The 'x + eps * (G(z) - x)' step, the dimention of eps is '[batch_size, h, w, c]' or 
'[batchsize, 1, 1, 1] and then broadcast to [batch_size, h, w, c]' ?

### 2. ACGAN in mnist with WGAN-GP loss
#### What is learned
- TODO: add noise to label. 给标签添加noise的作用应该是防止 `mode collapse` ,不加 noise 时ACGAN对相同的 condition 已经表现出了较好的利用 z 的 noise 来对抗 ` mode collapse`。
- 网络结构这一块就是试出来的，对 G 不加 z 和 label 的 embedding 会崩掉，bn 和 activation function 的顺序不对会崩掉，D 用 ln 替换 bn 会崩掉。也可能实际上是每个model对应的lr等超参数没调好。
- 注意loss和网络输出的关系，loss中注意label和logits的位置别弄反了...之前有个bug就是sigmoid算了两遍，结果loss变得有点奇怪...
- WGAN-GP提出来的目的就是为了更容易train GAN，所以WGAN-GP应该比DCGAN等更容易train，超参和模型更容易调整（效果更不更好倒不一定）。

#### Q
- WGAN-GP的理想loss变化是什么样的？

#### Result:
![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/AC-GAN(WGAN-GP%20loss).jpg)


### 3. WGAN-GP in mnist

#### Result:
![Alt text](https://github.com/1160300630/GANs-Playground/blob/master/images/WGAN-GP.jpg)
