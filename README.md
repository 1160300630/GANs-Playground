#### This is a simple GAN Zoo for studying

---

## Mostly using the Comics dataset

### 1. WGAN-GP
1. After resizing from 96 to 64 then the images are not belong to 0-255 any more, then doing the "img/127.5 + 1" process not exactly obey the thick transform img to -1~1, don't know weather it matters
2. The 'x + eps * (G(z) - x)' step, the dimention of eps is '[batch_size, h, w, c]' or 
'[batchsize, 1, 1, 1] and then broadcast to [batch_size, h, w, c]' ?

