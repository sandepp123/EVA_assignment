import matplotlib.pyplot as plt
import numpy as np
import torchvision

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
def plot_random(trainloader,classes):

	dataiter = iter(trainloader)
	images, labels = dataiter.next()

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
