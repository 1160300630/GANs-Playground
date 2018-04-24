import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_images(images, final=None, post_process=None):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if post_process is not None:
            plt.imshow(np.uint8((post_process(img.reshape([sqrtimg,sqrtimg])) + 1) * 127.5))
        else:
            plt.imshow(img.reshape([sqrtimg,sqrtimg]))
        if final is not None:
            plt.savefig(final + '.jpg')
    plt.show()
    return