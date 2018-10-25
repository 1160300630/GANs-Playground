import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_images_64_64(images, final=None, post_process=None):
    #images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    #sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

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
            plt.imshow(np.uint8(post_process(img)))
        else:
            plt.imshow(np.uint(img))
        if final is not None:
            plt.savefig(final + '.jpg')
    plt.show()
    return

def show_images_and_loss_64_64(images, D_losses=None, G_losses=None, final=None, post_process=None):
    #images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    #sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

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
            plt.imshow(np.uint8(post_process(img)))
        else:
            plt.imshow(np.uint(img))
        if final is not None:
            plt.savefig(final + '.jpg')
    plt.show()
    if D_losses is not None:
        plt.plot(range(len(D_losses)), D_losses)
        plt.title('D_loss')
        plt.show()
        D_losses.clear()
    if G_losses is not None:
        plt.plot(range(len(G_losses)), G_losses)
        plt.title('G_loss')
        plt.show()
        G_losses.clear()
    return


def show_images(images, final=None):
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
        # 28 x 28 need not to scale, -1~1 is fine
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
        if final is not None:
            plt.savefig(final + '.jpg')
    plt.show()
    return